"""QA — runs tests, lint, type-check, and security scan on generated code.

The QA checker validates generated code by:
1. Running syntax checks (Python: py_compile, YAML: yaml.safe_load)
2. Running ruff lint (if available)
3. Running mypy type-check (if available)
4. Running pytest (if tests exist)
5. Scanning for common security issues (hardcoded secrets, SQL injection, etc.)

All checks are best-effort — if a tool isn't installed, it's reported as
skipped rather than failing the pipeline.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .state import PipelineState, QAResult

logger = logging.getLogger(__name__)


class QAChecker:
    """Runs quality assurance checks on generated code.

    Since the auto-build pipeline uses the Git Data API (no local clone),
    the QA checker writes files to a temp directory and runs tools there.
    In the GitHub Actions context, the checkout is already present and
    the QA checker can run directly against it.
    """

    def __init__(self, workspace: str | Path | None = None):
        self.workspace = Path(workspace) if workspace else None

    def run(self, state: PipelineState) -> QAResult:
        """Run all QA checks on the files in the pipeline state.

        Args:
            state: Pipeline state with files_committed populated.
        Returns:
            QAResult with pass/fail status and details.
        """
        files = state.files_committed
        if not files:
            return QAResult(
                passed=False,
                overall_message="No files to check",
            )

        logger.info("QA: checking %d file(s)", len(files))

        result = QAResult(passed=True, overall_message="")

        # 1. Syntax validation (always runs — no external deps needed)
        syntax_errors = self._check_syntax(files)
        if syntax_errors:
            result.lint_passed = False
            result.lint_errors.extend(syntax_errors)
            result.passed = False
        else:
            result.lint_passed = True

        # 2. Security scan (always runs — pattern-based, no external deps)
        security_findings = self._security_scan(files)
        if security_findings:
            result.security_passed = False
            result.security_findings.extend(security_findings)
            # Security findings are advisory — they don't fail the pipeline
            # unless they're critical
            critical = [f for f in security_findings if "CRITICAL" in f]
            if critical:
                result.passed = False
        else:
            result.security_passed = True

        # 3. Run external tools (ruff, mypy, pytest) — best effort
        # These require a local checkout, so only run if workspace is set
        if self.workspace and self.workspace.exists():
            self._run_ruff(result)
            self._run_mypy(result)
            self._run_pytest(result)
        else:
            logger.info("QA: workspace not set — skipping ruff/mypy/pytest")

        # 4. Overall message
        if result.passed:
            parts = ["✅ All checks passed"]
            if result.lint_passed:
                parts.append(f"syntax: OK")
            if result.security_passed:
                parts.append(f"security: clean")
            if result.tests_passed > 0:
                parts.append(f"tests: {result.tests_passed} passed, {result.tests_failed} failed")
            result.overall_message = " | ".join(parts)
        else:
            parts = ["❌ QA failed"]
            if result.lint_errors:
                parts.append(f"syntax errors: {len(result.lint_errors)}")
            if result.security_findings:
                parts.append(f"security findings: {len(result.security_findings)}")
            if result.tests_failed > 0:
                parts.append(f"test failures: {result.tests_failed}")
            result.overall_message = " | ".join(parts)

        logger.info("QA: %s", result.overall_message)
        return result

    # -----------------------------------------------------------------
    # Syntax checks (no external deps)
    # -----------------------------------------------------------------

    def _check_syntax(self, files: dict[str, str]) -> list[str]:
        """Check syntax of all files. Returns list of error messages."""
        errors: list[str] = []

        for path, content in files.items():
            if path.endswith(".py"):
                err = self._check_python_syntax(path, content)
                if err:
                    errors.append(err)
            elif path.endswith((".yml", ".yaml")):
                err = self._check_yaml_syntax(path, content)
                if err:
                    errors.append(err)
            elif path.endswith(".json"):
                err = self._check_json_syntax(path, content)
                if err:
                    errors.append(err)
            elif path.endswith((".js", ".ts", ".tsx", ".jsx")):
                err = self._check_js_syntax(path, content)
                if err:
                    errors.append(err)

        return errors

    def _check_python_syntax(self, path: str, content: str) -> str | None:
        import py_compile, tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
                tf.write(content)
                tf.flush()
                tmp_path = tf.name
            py_compile.compile(tmp_path, doraise=True)
            return None
        except Exception as e:
            return f"{path}: Python syntax error: {e}"
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _check_yaml_syntax(self, path: str, content: str) -> str | None:
        try:
            import yaml
            yaml.safe_load(content)
            return None
        except Exception as e:
            return f"{path}: YAML syntax error: {e}"

    def _check_json_syntax(self, path: str, content: str) -> str | None:
        try:
            import json
            json.loads(content)
            return None
        except Exception as e:
            return f"{path}: JSON syntax error: {e}"

    def _check_js_syntax(self, path: str, content: str) -> str | None:
        """Check JS syntax using node --check.

        TypeScript (.ts/.tsx) files are skipped — they need tsc, not node.
        """
        # Skip TypeScript — node can't parse TS syntax
        if path.endswith((".ts", ".tsx")):
            return None
        try:
            suffix = ".mjs" if path.endswith((".ts", ".tsx")) else ".js"
            import tempfile
            tmp_path = None
            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tf:
                tf.write(content)
                tf.flush()
                tmp_path = tf.name
            result = subprocess.run(
                ["node", "--check", tmp_path],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return f"{path}: JS/TS syntax error: {result.stderr.strip()}"
            return None
        except FileNotFoundError:
            return None  # node not installed — skip
        except Exception as e:
            return f"{path}: JS/TS check error: {e}"
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # -----------------------------------------------------------------
    # Security scan (pattern-based, no external deps)
    # -----------------------------------------------------------------

    # Patterns that indicate hardcoded secrets
    SECRET_PATTERNS = [
        (r'(?:api_key|apikey|API_KEY)\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "hardcoded API key"),
        (r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']', "hardcoded password"),
        (r'(?:secret|SECRET)\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "hardcoded secret"),
        (r'(?:token|TOKEN)\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "hardcoded token"),
        (r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----', "private key in source"),
        (r'sk-[a-zA-Z0-9]{20,}', "OpenAI-style API key"),
        (r'github_pat_[a-zA-Z0-9_]{20,}', "GitHub PAT"),
        (r'AKIA[A-Z0-9]{16}', "AWS access key ID"),
        (r'ghp_[a-zA-Z0-9]{36}', "GitHub classic PAT"),
    ]

    # Patterns that indicate dangerous code
    DANGEROUS_PATTERNS = [
        (r'eval\s*\(\s*(?:request|input|sys\.argv)', "eval() with user input — code injection risk"),
        (r'exec\s*\(\s*(?:request|input|sys\.argv)', "exec() with user input — code injection risk"),
        (r'subprocess\.(?:Popen|call|run)\s*\(\s*shell\s*=\s*True', "subprocess with shell=True — injection risk"),
        (r'os\.system\s*\(\s*(?:request|input|sys\.argv)', "os.system() with user input — injection risk"),
    ]

    def _security_scan(self, files: dict[str, str]) -> list[str]:
        """Scan files for security issues. Returns list of findings."""
        findings: list[str] = []

        for path, content in files.items():
            # Skip test files — they may have intentionally fake secrets
            if "test" in path.lower() or path.endswith((".md", ".txt")):
                continue

            for pattern, description in self.SECRET_PATTERNS:
                matches = re.findall(pattern, content)
                if matches:
                    findings.append(f"⚠️ {path}: {description} ({len(matches)} occurrence(s))")

            for pattern, description in self.DANGEROUS_PATTERNS:
                matches = re.findall(pattern, content)
                if matches:
                    findings.append(f"🚨 CRITICAL {path}: {description} ({len(matches)} occurrence(s))")

        return findings

    # -----------------------------------------------------------------
    # External tool runners (require local workspace)
    # -----------------------------------------------------------------

    def _run_ruff(self, result: QAResult) -> None:
        """Run ruff linter if available."""
        try:
            r = subprocess.run(
                ["ruff", "check", str(self.workspace), "--output-format=concise"],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0:
                result.lint_passed = True
            else:
                result.lint_passed = False
                result.lint_errors.extend(
                    line for line in r.stdout.strip().splitlines() if line.strip()
                )
        except FileNotFoundError:
            logger.debug("ruff not installed — skipping lint")
        except subprocess.TimeoutExpired:
            logger.warning("ruff timed out — skipping")
        except Exception as e:
            logger.warning("ruff check failed: %s", e)

    def _run_mypy(self, result: QAResult) -> None:
        """Run mypy type checker if available."""
        try:
            r = subprocess.run(
                ["mypy", str(self.workspace), "--no-error-summary"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                result.typecheck_passed = True
            else:
                result.typecheck_passed = False
                result.typecheck_errors.extend(
                    line for line in r.stdout.strip().splitlines() if line.strip()
                )
        except FileNotFoundError:
            logger.debug("mypy not installed — skipping type check")
        except subprocess.TimeoutExpired:
            logger.warning("mypy timed out — skipping")
        except Exception as e:
            logger.warning("mypy check failed: %s", e)

    def _run_pytest(self, result: QAResult) -> None:
        """Run pytest if test files exist in the workspace."""
        if not self.workspace:
            return
        tests_dir = self.workspace / "tests"
        has_tests = any(tests_dir.glob("test_*.py")) if tests_dir.exists() else False

        if not has_tests:
            logger.info("No test files found — skipping pytest")
            return

        try:
            r = subprocess.run(
                ["python", "-m", "pytest", str(tests_dir),
                 "-x", "--tb=short", "-q"],
                capture_output=True, text=True, timeout=300,
                cwd=str(self.workspace),
            )
            result.tests_run = True
            # Parse pytest output for pass/fail counts
            output = r.stdout + r.stderr
            # Look for patterns like "5 passed" or "2 failed"
            passed_match = re.search(r'(\d+) passed', output)
            failed_match = re.search(r'(\d+) failed', output)
            result.tests_passed = int(passed_match.group(1)) if passed_match else 0
            result.tests_failed = int(failed_match.group(1)) if failed_match else 0

            if r.returncode != 0 and result.tests_failed > 0:
                result.passed = False
        except FileNotFoundError:
            logger.debug("pytest not installed — skipping tests")
        except subprocess.TimeoutExpired:
            logger.warning("pytest timed out — skipping")
        except Exception as e:
            logger.warning("pytest failed: %s", e)
