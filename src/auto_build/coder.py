"""Coder — executes a build plan by generating code and committing it.

The coder takes a BuildPlan and for each task:
1. Generates the file content using the LLM
2. Optionally reads existing file content (for modify actions)
3. Validates the generated code (basic syntax check)
4. Stages all files for a single commit
"""
from __future__ import annotations

import logging
import re
from typing import Any

from hermes_github.client import GitHubClient
from hermes_github.git_ops import GitOps

from .llm import LLMClient
from .state import BuildPlan, BuildTask, PipelineState

logger = logging.getLogger(__name__)


class Coder:
    """Executes build plans by generating code and committing to branches.

    The coder:
    1. Creates a feature branch via the GitHub API
    2. For each task in the plan, generates file content via the LLM
    3. Commits all files in a single commit via the Git Data API
    4. Returns the commit SHA and file contents
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        github: GitHubClient | None = None,
        git_ops: GitOps | None = None,
    ):
        self.llm = llm or LLMClient()
        self.github = github or GitHubClient()
        self.git_ops = git_ops or GitOps(self.github)
        self._owns_llm = llm is None
        self._owns_github = github is None

    def close(self) -> None:
        if self._owns_llm:
            self.llm.close()
        if self._owns_github:
            self.github.close()

    def __enter__(self) -> Coder:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def execute(self, state: PipelineState) -> tuple[str, dict[str, str]]:
        """Execute the build plan and return (commit_sha, files_dict).

        Args:
            state: Pipeline state with a populated plan.
        Returns:
            Tuple of (commit SHA, dict of file path → content).
        Raises:
            ValueError if no plan is set.
        """
        if state.plan is None:
            raise ValueError("Cannot execute: no plan in pipeline state")

        plan = state.plan
        owner = state.repo_owner
        repo = state.repo_name

        logger.info("Coder: executing plan with %d task(s) on %s/%s", len(plan.tasks), owner, repo)

        # 1. Create feature branch
        logger.info("Coder: creating branch %s from main", plan.branch_name)
        self.git_ops.create_feature_branch(owner, repo, plan.branch_name, base="main")

        # 2. Generate file contents for each task
        files: dict[str, str] = {}
        for i, task in enumerate(plan.tasks):
            logger.info("Coder: task %d/%d — %s (%s)", i + 1, len(plan.tasks), task.description, task.file_path)
            try:
                content = self._generate_file_content(task, owner, repo, plan)
                # Validate generated content
                self._validate_content(task.file_path, content)
                files[task.file_path] = content
                task.status = "completed"
                logger.info("Coder: task %d completed (%d chars)", i + 1, len(content))
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                logger.error("Coder: task %d failed: %s", i + 1, e)
                raise

        # 3. Commit all files in a single commit
        logger.info("Coder: committing %d file(s) to %s", len(files), plan.branch_name)
        commit_sha = self.git_ops.commit_files(
            owner, repo, plan.branch_name, files, plan.commit_message
        )

        logger.info("Coder: committed as %s", commit_sha[:7])
        return commit_sha, files

    def _generate_file_content(
        self,
        task: BuildTask,
        owner: str,
        repo: str,
        plan: BuildPlan,
    ) -> str:
        """Generate file content for a single task using the LLM."""
        # If modifying, read the existing file content
        existing_content = ""
        if task.action == "modify":
            existing_content = self.github.get_file_content(
                owner, repo, task.file_path, ref="main"
            ) or ""

        # Build the prompts
        system_prompt = self._system_prompt(task, plan)
        user_prompt = self._user_prompt(task, plan, existing_content)

        # Generate
        content = self.llm.generate_code(system_prompt, user_prompt, temperature=0.1)

        # Clean up any markdown fences the LLM might add
        content = self._strip_code_fences(content)

        return content

    def _system_prompt(self, task: BuildTask, plan: BuildPlan) -> str:
        """Build the system prompt for code generation."""
        return f"""You are a senior software engineer writing production-quality code.

Context:
- Repository: {plan.repo}
- Branch: {plan.branch_name}
- Plan summary: {plan.summary}
- Task: {task.description}
- File: {task.file_path}
- Action: {task.action}

Rules:
- Write complete, production-ready code — NO stubs, NO placeholders, NO TODOs
- Include proper error handling, type hints (Python), and docstrings
- Follow the existing code style of the project
- Include imports at the top of the file
- If writing tests, write real assertions with realistic test data
- If writing configuration, use real values or clearly marked environment variables
- Do NOT wrap the code in markdown code fences — output raw code only
"""

    def _user_prompt(self, task: BuildTask, plan: BuildPlan, existing_content: str) -> str:
        """Build the user prompt for code generation."""
        parts = [f"Generate the complete content for: {task.file_path}", ""]

        if task.content_preview:
            parts.append(f"Description: {task.description}")
            parts.append(f"Content guidance: {task.content_preview}")
            parts.append("")

        if existing_content:
            parts.append("Existing file content (modify in place):")
            parts.append("```")
            parts.append(existing_content[:5000])  # Cap at 5000 chars
            parts.append("```")
            parts.append("")
            parts.append("Produce the COMPLETE modified file. Include all unchanged parts.")
        else:
            parts.append("This is a new file. Produce its complete content.")

        parts.append("")
        parts.append(f"Plan summary for context: {plan.summary}")
        parts.append(f"PR title: {plan.pr_title}")

        return "\n".join(parts)

    def _strip_code_fences(self, content: str) -> str:
        """Remove markdown code fences if the LLM added them."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove the first line (fence with optional language)
            lines = lines[1:]
            # Remove the last line if it's a fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        return content.strip() + "\n"  # Ensure trailing newline

    def _validate_content(self, file_path: str, content: str) -> None:
        """Basic validation of generated file content.

        For Python files: check for syntax errors via py_compile.
        For other files: check for non-empty content.
        """
        if not content.strip():
            raise ValueError(f"Generated content for {file_path} is empty")

        if file_path.endswith(".py"):
            # Validate Python syntax
            import tempfile, py_compile, os
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(content)
                f.flush()
                try:
                    py_compile.compile(f.name, doraise=True)
                except py_compile.PyCompileError as e:
                    raise ValueError(
                        f"Generated Python code for {file_path} has syntax errors: {e}"
                    ) from e
                finally:
                    os.unlink(f.name)

        if file_path.endswith(".yml") or file_path.endswith(".yaml"):
            # Validate YAML
            import yaml
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Generated YAML for {file_path} has syntax errors: {e}"
                ) from e

        if file_path.endswith(".json"):
            # Validate JSON
            import json
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Generated JSON for {file_path} is invalid: {e}"
                ) from e
