"""Multi-repo workspace manager.

Handles cloning, syncing, and tracking multiple git repos on disk.
Designed for the multi-agent build pipeline where multiple repos
need to be worked on in coordinated fashion.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator

from .config import GitHubConfig
from .exceptions import GitError

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceRepo:
    """A single repo in the workspace."""
    name: str
    owner: str
    path: Path
    default_branch: str = "main"
    url: str = ""
    remote_configured: bool = False

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    def git(self, *args: str, check: bool = True) -> str:
        """Run a git command in this repo."""
        result = subprocess.run(
            ["git", *args],
            cwd=str(self.path),
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise GitError(
                f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout.strip()

    def current_branch(self) -> str:
        return self.git("rev-parse", "--abbrev-ref", "HEAD")

    def is_clean(self) -> bool:
        status = self.git("status", "--porcelain")
        return len(status.strip()) == 0

    def fetch(self, remote: str = "origin") -> None:
        self.git("fetch", remote)

    def checkout(self, branch: str) -> None:
        self.git("checkout", branch)

    def pull(self, remote: str = "origin", branch: str | None = None) -> None:
        args = ["pull", remote]
        if branch:
            args.append(branch)
        self.git(*args)

    def create_branch(self, name: str, from_branch: str | None = None) -> None:
        if from_branch:
            self.checkout(from_branch)
        self.git("checkout", "-b", name)

    def list_branches(self, remote: bool = False) -> list[str]:
        args = ["branch", "--list"]
        if remote:
            args.append("-r")
        output = self.git(*args)
        return [b.strip().lstrip("* ").strip() for b in output.splitlines() if b.strip()]


class Workspace:
    """Manages a collection of cloned repos under a root directory.

    Handles:
    - Cloning repos that don't exist locally
    - Syncing (fetch + pull) repos that do
    - Tracking repo metadata (owner, default branch, etc.)
    - Creating feature branches across repos
    - Cleaning up merged branches
    """

    def __init__(self, config: GitHubConfig | None = None):
        self.config = config or GitHubConfig.from_env()
        self.root: Path = self.config.workspace_root
        self.root.mkdir(parents=True, exist_ok=True)
        self._repos: dict[str, WorkspaceRepo] = {}

    @property
    def repos(self) -> dict[str, WorkspaceRepo]:
        """Dict of repo name → WorkspaceRepo."""
        return self._repos

    def add_repo(
        self,
        name: str,
        owner: str | None = None,
        default_branch: str = "main",
    ) -> WorkspaceRepo:
        """Register a repo in the workspace. Clones if not present."""
        owner = owner or self.config.default_owner
        url = f"https://github.com/{owner}/{name}.git"
        repo_path = self.root / name

        if not repo_path.exists():
            logger.info("Cloning %s/%s into %s", owner, name, repo_path)
            self._clone(url, repo_path)
        else:
            logger.debug("Repo %s already present at %s", name, repo_path)

        repo = WorkspaceRepo(
            name=name,
            owner=owner,
            path=repo_path,
            default_branch=default_branch,
            url=url,
            remote_configured=True,
        )
        self._repos[name] = repo
        return repo

    def _clone(self, url: str, path: Path) -> None:
        """Clone a repo from URL to path."""
        result = subprocess.run(
            ["git", "clone", url, str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise GitError(f"Failed to clone {url}: {result.stderr.strip()}")

    def sync(self, repo_name: str | None = None) -> None:
        """Fetch and pull for one or all repos.

        Args:
            repo_name: If specified, sync only that repo. Otherwise sync all.
        """
        repos = [self._repos[repo_name]] if repo_name else list(self._repos.values())
        for repo in repos:
            try:
                logger.info("Syncing %s...", repo.full_name)
                repo.fetch()
                current = repo.current_branch()
                if current and not current.startswith("origin/"):
                    repo.pull()
            except GitError as e:
                logger.warning("Failed to sync %s: %s", repo.full_name, e)

    def sync_all(self) -> None:
        """Sync all registered repos."""
        self.sync()

    def get_repo(self, name: str) -> WorkspaceRepo:
        """Get a registered repo by name."""
        if name not in self._repos:
            raise KeyError(f"Repo '{name}' not registered. Call add_repo() first.")
        return self._repos[name]

    def __contains__(self, name: str) -> bool:
        return name in self._repos

    def __iter__(self) -> Iterator[WorkspaceRepo]:
        return iter(self._repos.values())

    def __len__(self) -> int:
        return len(self._repos)

    def remove_repo(self, name: str) -> None:
        """Remove a repo from the workspace (deletes from disk)."""
        if name not in self._repos:
            return
        repo = self._repos.pop(name)
        if repo.path.exists():
            shutil.rmtree(repo.path)
        logger.info("Removed repo %s from workspace", name)

    def create_feature_branch(
        self,
        branch_name: str,
        repos: list[str] | None = None,
        from_branch: str = "main",
    ) -> dict[str, WorkspaceRepo]:
        """Create a feature branch across multiple repos.

        Args:
            branch_name: Name for the new branch.
            repos: List of repo names to branch. If None, branch all.
            from_branch: Base branch to branch from.

        Returns:
            Dict mapping repo name → WorkspaceRepo for repos where the branch was created.
        """
        target_repos = repos or list(self._repos.keys())
        result: dict[str, WorkspaceRepo] = {}

        for name in target_repos:
            repo = self._repos[name]
            try:
                # Ensure we're on the base branch and up to date
                repo.checkout(from_branch)
                repo.pull()
                repo.create_branch(branch_name, from_branch=from_branch)
                result[name] = repo
                logger.info("Created branch '%s' in %s", branch_name, name)
            except GitError as e:
                logger.warning("Failed to create branch in %s: %s", name, e)

        return result

    def cleanup_branch(self, branch_name: str, repos: list[str] | None = None) -> None:
        """Delete a branch from local repos (e.g., after merge)."""
        target_repos = repos or list(self._repos.keys())
        for name in target_repos:
            repo = self._repos[name]
            try:
                repo.checkout(repo.default_branch)
                repo.git("branch", "-D", branch_name, check=False)
                logger.info("Deleted branch '%s' from %s", branch_name, name)
            except GitError as e:
                logger.warning("Failed to delete branch in %s: %s", name, e)

    def status_report(self) -> dict[str, dict]:
        """Get git status for all repos (branch, clean/dirty, ahead/behind)."""
        report: dict[str, dict] = {}
        for name, repo in self._repos.items():
            try:
                branch = repo.current_branch()
                clean = repo.is_clean()
                ahead_behind = repo.git(
                    "rev-list", "--left-right", "--count",
                    f"origin/{repo.default_branch}...HEAD",
                    check=False,
                )
                behind, ahead = ahead_behind.split() if ahead_behind else ("0", "0")
                report[name] = {
                    "branch": branch,
                    "clean": clean,
                    "ahead": int(ahead),
                    "behind": int(behind),
                    "path": str(repo.path),
                }
            except GitError as e:
                report[name] = {"error": str(e)}
        return report
