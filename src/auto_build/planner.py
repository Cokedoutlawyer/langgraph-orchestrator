"""Planner — analyzes requirements and produces a structured build plan.

The planner takes an issue/requirement and uses an LLM to produce a
BuildPlan: what files to create/modify, what branch to use, commit message,
PR title and body. It also reads the existing repo structure to provide
context to the LLM.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from hermes_github.client import GitHubClient

from .llm import LLMClient
from .state import BuildPlan, BuildTask, PipelineState
from .web_fetcher import WebFetcher

logger = logging.getLogger(__name__)


class Planner:
    """Analyzes requirements and produces a BuildPlan.

    The planner:
    1. Reads the repo structure (file tree) via the GitHub API
    2. Reads key files (README, pyproject.toml, etc.) for context
    3. If the requirement mentions unfamiliar APIs/patterns, uses the
       WebFetcher to research them and add context for the LLM
    4. Sends the issue/requirement + context to the LLM
    5. Parses the LLM response into a BuildPlan
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        github: GitHubClient | None = None,
        web_fetcher: WebFetcher | None = None,
    ):
        self.llm = llm or LLMClient()
        self.github = github or GitHubClient()
        self.web_fetcher = web_fetcher or WebFetcher()
        self._owns_llm = llm is None
        self._owns_github = github is None
        self._owns_web = web_fetcher is None

    def close(self) -> None:
        if self._owns_llm:
            self.llm.close()
        if self._owns_github:
            self.github.close()
        if self._owns_web:
            self.web_fetcher.close()

    def __enter__(self) -> Planner:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def plan(self, state: PipelineState) -> BuildPlan:
        """Produce a BuildPlan from the pipeline state.

        Reads the repo structure, builds context, calls the LLM, and
        returns a structured plan. If the requirement mentions specific
        libraries, APIs, or patterns that may need research, the
        WebFetcher is used to gather additional context.
        """
        owner = state.repo_owner
        repo = state.repo_name

        logger.info("Planner: analyzing %s/%s for: %s", owner, repo, state.issue_title or state.requirements[:100])

        # 1. Gather repo context
        repo_info = self._get_repo_info(owner, repo)
        file_tree = self._get_file_tree(owner, repo)
        key_files = self._read_key_files(owner, repo)

        # 2. Research unfamiliar topics mentioned in the requirement
        research_context = self._research_requirement(state.requirements or state.issue_body or state.issue_title)

        # 3. Build the prompt
        system_prompt = self._system_prompt(repo_info, file_tree, key_files, research_context)
        user_prompt = self._user_prompt(state)

        # 3. Call LLM
        logger.info("Planner: calling LLM for plan generation...")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        plan_data = self.llm.chat_json(messages, temperature=0.2)

        # 4. Parse into BuildPlan
        plan = self._parse_plan(plan_data, owner, repo)
        logger.info(
            "Planner: produced plan with %d task(s), branch=%s",
            len(plan.tasks), plan.branch_name,
        )
        return plan

    def _get_repo_info(self, owner: str, repo: str) -> dict[str, Any]:
        """Get basic repo metadata."""
        try:
            r = self.github.get_repo(owner, repo)
            return {
                "name": r.name,
                "description": r.description or "",
                "default_branch": r.default_branch,
                "language": "",
            }
        except Exception as e:
            logger.warning("Failed to get repo info: %s", e)
            return {"name": repo, "description": "", "default_branch": "main", "language": ""}

    def _get_file_tree(self, owner: str, repo: str, max_depth: int = 3) -> str:
        """Get a compact file tree representation of the repo.

        Uses the GitHub API to get the tree and formats it as a
        human-readable tree string.
        """
        try:
            # Get the default branch's tree recursively
            data = self.github._request(
                "GET",
                f"/repos/{owner}/{repo}/git/trees/{self.github.get_default_branch(owner, repo)}?recursive=1",
            )
            tree = data.get("tree", [])

            # Filter and format
            lines: list[str] = []
            for item in tree:
                path = item["path"]
                # Skip hidden dirs, lock files, and very deep paths
                if any(part.startswith(".") and part not in (".github", ".env.example") for part in path.split("/")):
                    continue
                if path.endswith((".lock", ".pyc", ".so", ".png", ".jpg", ".woff2", ".ico")):
                    continue
                if path.startswith(("node_modules/", ".git/", "__pycache__/")):
                    continue
                depth = path.count("/")
                if depth > max_depth:
                    continue
                prefix = "  " * depth
                icon = "📁" if item["type"] == "tree" else "📄"
                lines.append(f"{prefix}{icon} {path}")

            return "\n".join(lines[:500])  # Cap at 500 lines
        except Exception as e:
            logger.warning("Failed to get file tree: %s", e)
            return "(file tree unavailable)"

    def _read_key_files(self, owner: str, repo: str) -> dict[str, str]:
        """Read key files (README, pyproject.toml, package.json) for context."""
        key_file_names = [
            "README.md",
            "pyproject.toml",
            "package.json",
            "AGENTS.md",
            "CONTRIBUTING.md",
            "src/__init__.py",
        ]
        contents: dict[str, str] = {}
        for path in key_file_names:
            content = self.github.get_file_content(owner, repo, path)
            if content:
                # Truncate to keep context manageable
                contents[path] = content[:3000]
        return contents

    def _system_prompt(self, repo_info: dict, file_tree: str, key_files: dict, research_context: str = "") -> str:
        """Build the system prompt with repo context."""
        key_files_str = ""
        for path, content in key_files.items():
            key_files_str += f"\n--- {path} ---\n{content[:2000]}\n"

        research_str = ""
        if research_context:
            research_str = f"\n## Additional Research Context\n{research_context}\n"

        return f"""You are a senior software architect planning a code change for a GitHub repository.

Repository: {repo_info['name']}
Description: {repo_info['description']}
Default branch: {repo_info['default_branch']}

Current file tree:
{file_tree}

Key file contents:
{key_files_str}
{research_str}
Your job: analyze the requirement and produce a detailed, actionable build plan.

Output a JSON object with these fields:
- "summary": One-paragraph summary of what will be built/changed
- "branch_name": A git branch name (use convention: auto-build/<short-slug>)
- "commit_message": A conventional-commits style commit message
- "pr_title": A clear PR title (max 72 chars)
- "pr_body": PR body with ## Summary, ## Changes, ## Testing sections
- "tasks": Array of task objects, each with:
  - "description": What this task does
  - "file_path": Path to the file to create or modify
  - "action": "create" or "modify"
  - "content_preview": Brief description of what the file should contain

Rules:
- Be specific about file paths — they must match the existing structure
- Every task must be actionable — no vague "refactor X" tasks
- Prefer creating new files over modifying existing ones when possible
- Include tests in the plan if the repo has a test framework
- No placeholders, no TODOs, no stubs — every task must produce real, complete code
- Branch name must be kebab-case and start with "auto-build/"
"""

    def _user_prompt(self, state: PipelineState) -> str:
        """Build the user prompt with the requirement details."""
        parts = ["Please plan the following change:\n"]

        if state.issue_title:
            parts.append(f"## Issue: {state.issue_title}")
        if state.issue_body:
            parts.append(f"\n{state.issue_body}")
        if state.requirements:
            parts.append(f"\n## Requirements\n{state.requirements}")

        parts.append(f"\n## Target Repository\n{state.repo_owner}/{state.repo_name}")

        return "\n".join(parts)

    def _research_requirement(self, text: str) -> str:
        """Research unfamiliar topics mentioned in the requirement.

        Extracts potential research topics from the text (library names,
        API names, framework patterns) and uses the WebFetcher to find
        relevant documentation. Returns a combined research context string.
        """
        if not text:
            return ""

        # Extract potential research topics
        topics = self._extract_research_topics(text)
        if not topics:
            return ""

        logger.info("Planner: researching %d topic(s): %s", len(topics), ", ".join(topics))

        research_parts: list[str] = []
        for topic in topics[:3]:  # Limit to 3 research queries to stay fast
            try:
                result = self.web_fetcher.research(topic, max_results=2, max_chars_per_page=2000)
                if result and "no web results" not in result.lower():
                    research_parts.append(result)
            except Exception as e:
                logger.warning("Planner: research failed for '%s': %s", topic, e)

        return "\n\n".join(research_parts) if research_parts else ""

    def _extract_research_topics(self, text: str) -> list[str]:
        """Extract potential research topics from a text.

        Looks for:
        - Library/framework names (e.g., "FastAPI", "React", "PyTorch")
        - API names (e.g., "OpenAI API", "GitHub Actions")
        - Pattern mentions (e.g., "dependency injection", "factory pattern")
        - URLs
        """
        import re

        topics: list[str] = []

        # Known framework/library names (case-sensitive)
        known_libs = [
            "FastAPI", "Flask", "Django", "Pydantic", "LangChain", "LangGraph",
            "React", "Vue", "Angular", "Next.js", "Nuxt",
            "PyTorch", "TensorFlow", "JAX", "Transformers",
            "httpx", "aiohttp", "requests", "tenacity",
            "pytest", "ruff", "mypy", "pyright",
            "Docker", "Kubernetes", "Helm",
            "GitHub Actions", "terraform",
            "OpenAI", "Anthropic", "Gemini", "Claude",
        ]

        text_lower = text.lower()
        for lib in known_libs:
            if lib.lower() in text_lower:
                topics.append(f"{lib} documentation and usage pattern")

        # Look for "how to X" or "what is X" patterns
        how_to = re.findall(r"(?:how to|what is|what are)\s+([a-z\s]{5,40})", text_lower)
        for match in how_to[:2]:
            topics.append(match.strip())

        # Look for API endpoint patterns
        api_mentions = re.findall(r"(?:GET|POST|PUT|DELETE|PATCH)\s+/[a-z/_-]+", text, re.IGNORECASE)
        for match in api_mentions[:2]:
            topics.append(f"REST API {match}")

        # Deduplicate and limit
        seen = set()
        unique: list[str] = []
        for t in topics:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)

        return unique[:5]

    def _parse_plan(self, data: dict, owner: str, repo: str) -> BuildPlan:
        """Parse the LLM JSON response into a BuildPlan."""
        tasks = []
        for t in data.get("tasks", []):
            tasks.append(BuildTask(
                description=t["description"],
                file_path=t["file_path"],
                action=t.get("action", "create"),
                content_preview=t.get("content_preview", ""),
            ))

        branch_name = data.get("branch_name", "auto-build/untitled")
        # Ensure branch name follows convention
        if not branch_name.startswith("auto-build/"):
            branch_name = f"auto-build/{branch_name}"
        # Sanitize branch name
        branch_name = re.sub(r"[^a-zA-Z0-9/_-]", "-", branch_name)

        return BuildPlan(
            summary=data.get("summary", ""),
            repo=repo,
            branch_name=branch_name,
            commit_message=data.get("commit_message", "feat: auto-build change"),
            pr_title=data.get("pr_title", "Auto-build change"),
            pr_body=data.get("pr_body", "## Summary\nAuto-generated by Hermes auto-build pipeline."),
            tasks=tasks,
            estimated_files=len(tasks),
        )
