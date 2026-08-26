"""Integration tests for hermes_github.GitHubClient.

All tests use unittest.mock to mock httpx responses — no network access required.
Covers: construction, get_repo, create_branch, list_prs, merge_pr,
commit_multiple_files (Git Data API sequence), and error handling.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import httpx
import pytest

from hermes_github.client import GitHubClient
from hermes_github.config import GitHubConfig
from hermes_github.exceptions import (
    GitHubAPIError,
    GitHubAuthError,
    GitHubNotFoundError,
    GitHubRateLimitError,
)
from hermes_github.models import (
    Branch,
    MergeMethod,
    PullRequest,
    Repo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_response(
    json_body: object = None,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    *,
    content: bytes | None = None,
) -> httpx.Response:
    """Build a mock httpx.Response with realistic headers."""
    hdrs = {
        "x-ratelimit-remaining": "5000",
        "x-ratelimit-reset": "9999999999",
    }
    if headers:
        hdrs.update(headers)
    if content is not None:
        resp = httpx.Response(
            status_code=status_code,
            headers=hdrs,
            content=content,
        )
    else:
        resp = httpx.Response(
            status_code=status_code,
            headers=hdrs,
            json=json_body,
        )
    return resp


def make_client(
    config: GitHubConfig | None = None,
    responses: list[httpx.Response] | httpx.Response | None = None,
) -> GitHubClient:
    """Create a GitHubClient whose httpx client.request is mocked.

    *responses* may be a single Response (returned for every call) or a list
    (returned in order, repeating the last element for extra calls).

    Records each call's params as a snapshot copy in ``client._call_log``
    so tests can inspect pagination params that the client mutates in place.
    """
    cfg = config or GitHubConfig(
        token="test-token",
        default_owner="octocat",
        max_retries=3,
        retry_base_delay=0.0,  # keep tests fast
        rate_limit_buffer=0,   # never trigger proactive rate-limit sleep
    )
    client = GitHubClient(cfg)

    if responses is None:
        responses = [make_response(json_body={})]

    if isinstance(responses, httpx.Response):
        responses = [responses]

    mock_http = MagicMock(spec=httpx.Client)
    client._client = mock_http
    client._call_log = []  # snapshots of (method, url, params_copy)

    response_iter = iter(responses)
    last = responses[-1]

    def _side_effect(method, url, params=None, json=None):
        # Snapshot params at call time (paginate_iterator mutates the dict in place)
        client._call_log.append((method, url, dict(params) if params else None))
        try:
            return next(response_iter)
        except StopIteration:
            return last

    mock_http.request.side_effect = _side_effect
    return client


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestGitHubClientConstruction:
    """Test GitHubClient construction with config."""

    def test_construct_with_explicit_config(self):
        cfg = GitHubConfig(
            token="my-token",
            default_owner="my-org",
            api_base_url="https://api.github.com",
            request_timeout=42.0,
        )
        client = GitHubClient(cfg)
        assert client.config is cfg
        assert client.config.token_str == "my-token"
        assert client.config.default_owner == "my-org"

    def test_client_lazy_init_creates_httpx_client(self):
        """The httpx.Client should only be created on first access."""
        cfg = GitHubConfig(token="tok")
        client = GitHubClient(cfg)
        assert client._client is None
        # Accessing .client property triggers lazy init
        http = client.client
        assert isinstance(http, httpx.Client)
        assert client._client is http
        # base_url and headers from config
        assert str(http.base_url).rstrip("/") == cfg.api_base_url
        assert http.headers["Authorization"] == "Bearer tok"
        client.close()
        assert client._client is None

    def test_context_manager_closes_client(self):
        cfg = GitHubConfig(token="tok")
        with GitHubClient(cfg) as client:
            _ = client.client  # force creation
            assert client._client is not None
        assert client._client is None

    def test_headers_contain_auth_and_api_version(self):
        cfg = GitHubConfig(token="abc123", api_version="2022-11-28")
        h = cfg.headers()
        assert h["Authorization"] == "Bearer abc123"
        assert h["Accept"] == "application/vnd.github+json"
        assert h["X-GitHub-Api-Version"] == "2022-11-28"
        assert "User-Agent" in h


# ---------------------------------------------------------------------------
# 2. get_repo
# ---------------------------------------------------------------------------

class TestGetRepo:
    """Test get_repo parses API response correctly."""

    def test_get_repo_parses_response(self):
        repo_api = {
            "id": 12345,
            "name": "hello-world",
            "full_name": "octocat/hello-world",
            "owner": {"login": "octocat"},
            "private": False,
            "default_branch": "main",
            "clone_url": "https://github.com/octocat/hello-world.git",
            "html_url": "https://github.com/octocat/hello-world",
            "description": "My first repo",
            "archived": False,
            "disabled": False,
        }
        client = make_client(responses=make_response(json_body=repo_api))
        repo = client.get_repo("octocat", "hello-world")

        assert isinstance(repo, Repo)
        assert repo.id == 12345
        assert repo.name == "hello-world"
        assert repo.full_name == "octocat/hello-world"
        assert repo.owner == "octocat"
        assert repo.default_branch == "main"
        assert repo.clone_url == "https://github.com/octocat/hello-world.git"

        # Verify correct method + URL
        mock_http = client._client
        mock_http.request.assert_called_once()
        actual_call = mock_http.request.call_args
        assert actual_call.args[0] == "GET"
        assert actual_call.args[1] == "/repos/octocat/hello-world"

    def test_get_repo_default_branch_field(self):
        repo_api = {
            "id": 1,
            "name": "r",
            "full_name": "o/r",
            "owner": {"login": "o"},
            "default_branch": "develop",
        }
        client = make_client(responses=make_response(json_body=repo_api))
        repo = client.get_repo("o", "r")
        assert repo.default_branch == "develop"


# ---------------------------------------------------------------------------
# 3. create_branch
# ---------------------------------------------------------------------------

class TestCreateBranch:
    """Test create_branch sends correct payload."""

    def test_create_branch_sends_correct_payload(self):
        # First call: GET /repos/.../branches/main  → branch data with SHA
        # Second call: POST /repos/.../git/refs     → create ref
        branch_resp = make_response(json_body={
            "name": "main",
            "commit": {"sha": "abc123source", "url": "..."},
            "protected": False,
        })
        ref_resp = make_response(json_body={
            "ref": "refs/heads/feature",
            "node_id": "REF123",
            "url": "https://api.github.com/repos/octocat/hello-world/git/refs/heads/feature",
            "object": {"sha": "abc123source", "type": "commit"},
        })
        client = make_client(responses=[branch_resp, ref_resp])
        result = client.create_branch(
            "octocat", "hello-world", "feature", from_branch="main"
        )

        assert result.branch_name == "feature"
        assert result.sha == "abc123source"
        assert result.repo_full_name == "octocat/hello-world"

        mock_http = client._client
        assert mock_http.request.call_count == 2

        # First call: GET branch to get SHA
        first = mock_http.request.call_args_list[0]
        assert first.args[0] == "GET"
        assert first.args[1] == "/repos/octocat/hello-world/branches/main"

        # Second call: POST ref with correct payload
        second = mock_http.request.call_args_list[1]
        assert second.args[0] == "POST"
        assert second.args[1] == "/repos/octocat/hello-world/git/refs"
        assert second.kwargs["json"] == {
            "ref": "refs/heads/feature",
            "sha": "abc123source",
        }

    def test_create_branch_custom_source(self):
        """create_branch from a non-main branch uses that branch's SHA."""
        branch_resp = make_response(json_body={
            "name": "develop",
            "commit": {"sha": "devsha999"},
            "protected": False,
        })
        ref_resp = make_response(json_body={"ref": "refs/heads/feature2"})
        client = make_client(responses=[branch_resp, ref_resp])
        result = client.create_branch(
            "octocat", "hello-world", "feature2", from_branch="develop"
        )
        assert result.sha == "devsha999"

        # The ref payload should reference the dev SHA
        second_call = client._client.request.call_args_list[1]
        assert second_call.kwargs["json"]["sha"] == "devsha999"


# ---------------------------------------------------------------------------
# 4. list_prs — pagination
# ---------------------------------------------------------------------------

class TestListPRs:
    """Test list_prs parses pagination."""

    def _pr_data(self, num: int, state: str = "open") -> dict:
        return {
            "id": 1000 + num,
            "number": num,
            "title": f"PR #{num}",
            "body": f"Body of PR {num}",
            "state": state,
            "draft": False,
            "merged": False,
            "mergeable": True,
            "head": {"ref": "feature", "sha": f"sha{num}"},
            "base": {"ref": "main", "sha": "basesha"},
            "user": {"login": "octocat"},
            "labels": [],
            "requested_reviewers": [],
            "html_url": f"https://github.com/octocat/hello-world/pull/{num}",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        }

    def test_list_prs_single_page(self):
        """Fewer than per_page results → single page, no second request."""
        prs = [self._pr_data(i) for i in range(1, 4)]
        client = make_client(responses=make_response(json_body=prs))
        results = client.list_prs("octocat", "hello-world")

        assert len(results) == 3
        assert all(isinstance(p, PullRequest) for p in results)
        assert results[0].number == 1
        assert results[2].number == 3

        # Only one page requested
        assert client._client.request.call_count == 1
        first = client._client.request.call_args_list[0]
        assert first.args[0] == "GET"
        assert first.args[1] == "/repos/octocat/hello-world/pulls"
        # params include state, sort, direction, per_page, page
        params = first.kwargs["params"]
        assert params["state"] == "open"
        assert params["page"] == 1

    def test_list_prs_multi_page(self):
        """More than per_page results → pagination across multiple pages.

        list_prs uses paginate() with default per_page=100, so page1 must
        have exactly 100 items to trigger a second-page request.
        """
        page1 = [self._pr_data(i) for i in range(1, 101)]   # 100 items = per_page
        page2 = [self._pr_data(i) for i in range(101, 103)] # 2 items < per_page → last page
        client = make_client(
            responses=[make_response(json_body=page1), make_response(json_body=page2)]
        )
        results = client.list_prs("octocat", "hello-world")

        assert len(results) == 102
        assert results[0].number == 1
        assert results[100].number == 101
        assert results[101].number == 102

        # Two pages requested
        assert client._client.request.call_count == 2
        # Use _call_log snapshots — paginate_iterator mutates the params dict
        # in place, so call_args would show the final state for every call.
        first_method, first_url, first_params = client._call_log[0]
        second_method, second_url, second_params = client._call_log[1]
        assert first_params["page"] == 1
        assert second_params["page"] == 2
        assert first_params["per_page"] == 100

    def test_paginate_multi_page_small_per_page(self):
        """Test pagination mechanism directly with a small per_page value."""
        page1 = [self._pr_data(i) for i in range(1, 6)]   # 5 items = per_page
        page2 = [self._pr_data(i) for i in range(6, 8)]   # 2 items < per_page → last page
        client = make_client(
            responses=[make_response(json_body=page1), make_response(json_body=page2)]
        )
        results = client.paginate(
            "GET", "/repos/octocat/hello-world/pulls", per_page=5
        )
        assert len(results) == 7
        assert client._client.request.call_count == 2

    def test_list_prs_empty(self):
        """No PRs → empty list."""
        client = make_client(responses=make_response(json_body=[]))
        results = client.list_prs("octocat", "hello-world")
        assert results == []

    def test_list_prs_state_param(self):
        """list_prs passes state param through to API."""
        client = make_client(responses=make_response(json_body=[]))
        client.list_prs("octocat", "hello-world", state="closed")
        params = client._client.request.call_args.kwargs["params"]
        assert params["state"] == "closed"


# ---------------------------------------------------------------------------
# 5. merge_pr
# ---------------------------------------------------------------------------

class TestMergePR:
    """Test merge_pr sends correct method and title."""

    def test_merge_pr_squash_default(self):
        merge_resp = make_response(json_body={
            "sha": "mergedsha123",
            "merged": True,
            "message": "Pull request successfully merged",
        })
        client = make_client(responses=merge_resp)
        result = client.merge_pr("octocat", "hello-world", 42)

        assert result.merged is True
        assert result.sha == "mergedsha123"
        assert "merged" in result.message.lower()

        mock_http = client._client
        mock_http.request.assert_called_once()
        c = mock_http.request.call_args
        assert c.args[0] == "PUT"
        assert c.args[1] == "/repos/octocat/hello-world/pulls/42/merge"
        assert c.kwargs["json"]["merge_method"] == "squash"

    def test_merge_pr_with_title_and_message(self):
        merge_resp = make_response(json_body={
            "sha": "sha456",
            "merged": True,
            "message": "Merged",
        })
        client = make_client(responses=merge_resp)
        result = client.merge_pr(
            "octocat", "hello-world", 99,
            method=MergeMethod.MERGE,
            commit_title="Custom Merge Title (ref #99)",
            commit_message="Custom body text",
        )

        assert result.merged is True
        c = client._client.request.call_args
        assert c.args[0] == "PUT"
        assert c.kwargs["json"]["merge_method"] == "merge"
        assert c.kwargs["json"]["commit_title"] == "Custom Merge Title (ref #99)"
        assert c.kwargs["json"]["commit_message"] == "Custom body text"

    def test_merge_pr_rebase_method(self):
        merge_resp = make_response(json_body={"sha": "sha", "merged": True, "message": "ok"})
        client = make_client(responses=merge_resp)
        client.merge_pr("o", "r", 1, method=MergeMethod.REBASE)
        assert client._client.request.call_args.kwargs["json"]["merge_method"] == "rebase"

    def test_merge_pr_api_error_returns_not_merged(self):
        """When the API returns a non-2xx, merge_pr catches GitHubAPIError and returns merged=False."""
        # 422 — not mergeable, treated as generic 4xx GitHubAPIError
        err_resp = make_response(
            json_body={"message": "Pull Request is not mergeable"},
            status_code=422,
        )
        client = make_client(responses=err_resp)
        result = client.merge_pr("octocat", "hello-world", 42)
        assert result.merged is False


# ---------------------------------------------------------------------------
# 6. commit_multiple_files — Git Data API sequence
# ---------------------------------------------------------------------------

class TestCommitMultipleFiles:
    """Test commit_multiple_files via the blob→tree→commit→ref sequence."""

    def test_commit_multiple_files_full_sequence(self):
        """Verify the complete Git Data API call sequence and payload correctness."""
        # Responses in order:
        # 1. GET /repos/.../branches/{branch} → branch HEAD
        # 2. GET /repos/.../git/trees/{sha}   → base tree
        # 3. POST /repos/.../git/blobs        → blob SHA (called once per file)
        # 4. POST /repos/.../git/trees        → new tree SHA
        # 5. POST /repos/.../git/commits      → new commit SHA
        # 6. PATCH /repos/.../git/refs/heads/{branch} → update ref

        branch_resp = make_response(json_body={
            "name": "main",
            "commit": {"sha": "parentcommitsha"},
            "protected": False,
        })
        tree_resp = make_response(json_body={
            "sha": "basetreesha",
            "tree": [{"path": "README.md", "mode": "100644", "type": "blob", "sha": "oldsha"}],
            "truncated": False,
        })
        blob1_resp = make_response(json_body={"sha": "blobsha1", "url": "..."})
        blob2_resp = make_response(json_body={"sha": "blobsha2", "url": "..."})
        new_tree_resp = make_response(json_body={"sha": "newtreesha", "url": "..."})
        new_commit_resp = make_response(json_body={"sha": "newcommitsha", "url": "..."})
        ref_resp = make_response(json_body={"sha": "newcommitsha", "ref": "refs/heads/main"})

        client = make_client(responses=[
            branch_resp,       # get_branch
            tree_resp,         # get_tree
            blob1_resp,        # create_blob file1
            blob2_resp,        # create_blob file2
            new_tree_resp,     # create_tree
            new_commit_resp,   # create_commit
            ref_resp,          # update_ref
        ])

        result = client.commit_multiple_files(
            "octocat", "hello-world", "main",
            files={"file1.txt": "content1", "file2.txt": "content2"},
            message="Add two files",
        )

        # Should return the new commit SHA
        assert result == "newcommitsha"

        mock_http = client._client
        assert mock_http.request.call_count == 7
        calls = mock_http.request.call_args_list

        # 1. GET branch
        assert calls[0].args[0] == "GET"
        assert calls[0].args[1] == "/repos/octocat/hello-world/branches/main"

        # 2. GET tree
        assert calls[1].args[0] == "GET"
        assert calls[1].args[1] == "/repos/octocat/hello-world/git/trees/parentcommitsha"

        # 3-4. POST blobs (one per file)
        blob_calls = [calls[2], calls[3]]
        blob_contents = set()
        for bc in blob_calls:
            assert bc.args[0] == "POST"
            assert bc.args[1] == "/repos/octocat/hello-world/git/blobs"
            assert bc.kwargs["json"]["encoding"] == "utf-8"
            blob_contents.add(bc.kwargs["json"]["content"])
        # The two blob payloads should have different content
        assert blob_contents == {"content1", "content2"}

        # 5. POST tree — should include base_tree and entries for both files
        tree_call = calls[4]
        assert tree_call.args[0] == "POST"
        assert tree_call.args[1] == "/repos/octocat/hello-world/git/trees"
        tree_payload = tree_call.kwargs["json"]
        assert tree_payload["base_tree"] == "basetreesha"
        assert len(tree_payload["tree"]) == 2
        # Verify each entry has correct mode/type
        for entry in tree_payload["tree"]:
            assert entry["mode"] == "100644"
            assert entry["type"] == "blob"
            assert entry["path"] in ("file1.txt", "file2.txt")
        # The entry SHAs should match blob SHAs
        entry_shas = {e["sha"] for e in tree_payload["tree"]}
        assert entry_shas == {"blobsha1", "blobsha2"}

        # 6. POST commit
        commit_call = calls[5]
        assert commit_call.args[0] == "POST"
        assert commit_call.args[1] == "/repos/octocat/hello-world/git/commits"
        commit_payload = commit_call.kwargs["json"]
        assert commit_payload["tree"] == "newtreesha"
        assert commit_payload["parents"] == ["parentcommitsha"]
        assert commit_payload["message"] == "Add two files"

        # 7. PATCH ref
        ref_call = calls[6]
        assert ref_call.args[0] == "PATCH"
        assert ref_call.args[1] == "/repos/octocat/hello-world/git/refs/heads/main"
        assert ref_call.kwargs["json"] == {"sha": "newcommitsha"}

    def test_commit_single_file(self):
        """Single file commit still goes through the full Git Data sequence."""
        branch_resp = make_response(json_body={
            "name": "main", "commit": {"sha": "psha"}, "protected": False,
        })
        tree_resp = make_response(json_body={"sha": "btsha", "tree": [], "truncated": False})
        blob_resp = make_response(json_body={"sha": "bsha1"})
        new_tree_resp = make_response(json_body={"sha": "ntsha"})
        new_commit_resp = make_response(json_body={"sha": "ncsha"})
        ref_resp = make_response(json_body={"sha": "ncsha"})

        client = make_client(responses=[
            branch_resp, tree_resp, blob_resp, new_tree_resp, new_commit_resp, ref_resp,
        ])
        result = client.commit_multiple_files(
            "o", "r", "main",
            files={"only.txt": "data"},
            message="single file",
        )
        assert result == "ncsha"
        assert client._client.request.call_count == 6

        # Verify tree payload has exactly one entry
        tree_call = client._client.request.call_args_list[3]
        tree_payload = tree_call.kwargs["json"]
        assert len(tree_payload["tree"]) == 1
        assert tree_payload["tree"][0]["path"] == "only.txt"

    def test_commit_multiple_files_no_base_tree(self):
        """When get_tree returns a dict without 'sha', base_tree is None."""
        branch_resp = make_response(json_body={
            "name": "main", "commit": {"sha": "psha"}, "protected": False,
        })
        # tree_data.get("sha") → None
        tree_resp = make_response(json_body={"tree": [], "truncated": False})
        blob_resp = make_response(json_body={"sha": "bsha"})
        new_tree_resp = make_response(json_body={"sha": "ntsha"})
        new_commit_resp = make_response(json_body={"sha": "ncsha"})
        ref_resp = make_response(json_body={"sha": "ncsha"})

        client = make_client(responses=[
            branch_resp, tree_resp, blob_resp, new_tree_resp, new_commit_resp, ref_resp,
        ])
        result = client.commit_multiple_files("o", "r", "main", {"f": "c"}, "m")
        assert result == "ncsha"

        # create_tree should NOT include base_tree key
        tree_call = client._client.request.call_args_list[3]
        tree_payload = tree_call.kwargs["json"]
        assert "base_tree" not in tree_payload


# ---------------------------------------------------------------------------
# 7. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test that HTTP error codes raise the correct exceptions."""

    def test_401_raises_auth_error(self):
        err_resp = make_response(
            json_body={"message": "Bad credentials"},
            status_code=401,
        )
        client = make_client(responses=err_resp)
        with pytest.raises(GitHubAuthError):
            client.get_repo("octocat", "hello-world")

    def test_404_raises_not_found_error(self):
        err_resp = make_response(
            json_body={"message": "Not Found"},
            status_code=404,
        )
        client = make_client(responses=err_resp)
        with pytest.raises(GitHubNotFoundError):
            client.get_repo("octocat", "nonexistent-repo")

    def test_403_with_rate_limit_remaining_zero_raises_rate_limit_error(self):
        """403 with x-ratelimit-remaining: 0 → GitHubRateLimitError."""
        err_resp = make_response(
            json_body={"message": "API rate limit exceeded"},
            status_code=403,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "9999999999"},
        )
        client = make_client(responses=err_resp)
        with pytest.raises(GitHubRateLimitError) as exc_info:
            client.get_repo("octocat", "hello-world")
        assert exc_info.value.status_code == 403

    def test_403_without_rate_limit_raises_auth_error(self):
        """403 without rate-limit-remaining=0 → GitHubAuthError (permissions)."""
        err_resp = make_response(
            json_body={"message": "Resource not accessible by integration"},
            status_code=403,
            headers={"x-ratelimit-remaining": "5000"},
        )
        client = make_client(responses=err_resp)
        with pytest.raises(GitHubAuthError):
            client.get_repo("octocat", "hello-world")

    def test_422_raises_generic_api_error(self):
        """422 (validation) → GitHubAPIError."""
        err_resp = make_response(
            json_body={"message": "Validation Failed"},
            status_code=422,
        )
        client = make_client(responses=err_resp)
        with pytest.raises(GitHubAPIError) as exc_info:
            client.create_pr("octocat", "hello-world", "title", "head")
        assert exc_info.value.status_code == 422

    def test_500_retries_then_raises_on_exhaustion(self):
        """500 server errors are retried; after max_retries, GitHubAPIError is raised."""
        cfg = GitHubConfig(
            token="t",
            max_retries=2,
            retry_base_delay=0.0,
            rate_limit_buffer=0,
        )
        err_resp = make_response(
            json_body={"message": "Server Error"},
            status_code=500,
        )
        client = make_client(config=cfg, responses=err_resp)
        with pytest.raises(GitHubAPIError):
            client.get_repo("octocat", "hello-world")
        # Should have been called max_retries times
        assert client._client.request.call_count == 2

    def test_500_retries_then_succeeds(self):
        """500 on first attempt, 200 on second → success."""
        cfg = GitHubConfig(
            token="t",
            max_retries=3,
            retry_base_delay=0.0,
            rate_limit_buffer=0,
        )
        repo_api = {
            "id": 1, "name": "r", "full_name": "o/r",
            "owner": {"login": "o"}, "default_branch": "main",
        }
        client = make_client(
            config=cfg,
            responses=[
                make_response(json_body={"message": "Server Error"}, status_code=500),
                make_response(json_body=repo_api, status_code=200),
            ],
        )
        repo = client.get_repo("o", "r")
        assert repo.name == "r"
        assert client._client.request.call_count == 2

    def test_404_on_get_file_content_returns_none(self):
        """get_file_content catches GitHubNotFoundError and returns None."""
        err_resp = make_response(
            json_body={"message": "Not Found"},
            status_code=404,
        )
        client = make_client(responses=err_resp)
        result = client.get_file_content("octocat", "hello-world", "missing.txt")
        assert result is None

    def test_network_error_retries(self):
        """httpx.NetworkError is retried like a 5xx."""
        cfg = GitHubConfig(
            token="t",
            max_retries=2,
            retry_base_delay=0.0,
            rate_limit_buffer=0,
        )
        repo_api = {
            "id": 1, "name": "r", "full_name": "o/r",
            "owner": {"login": "o"}, "default_branch": "main",
        }
        client = make_client(
            config=cfg,
            responses=[
                httpx.ConnectError("Connection refused"),
                make_response(json_body=repo_api, status_code=200),
            ],
        )
        # The side_effect in make_client uses next(iter), so we need to
        # handle the exception type. Rebuild the mock to raise on first call.
        mock_http = client._client
        repo_resp = make_response(json_body=repo_api, status_code=200)
        call_log = [httpx.ConnectError("Connection refused"), repo_resp]

        def _side_effect(method, url, params=None, json=None):
            val = call_log.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        mock_http.request.side_effect = _side_effect

        repo = client.get_repo("o", "r")
        assert repo.name == "r"
        assert mock_http.request.call_count == 2
