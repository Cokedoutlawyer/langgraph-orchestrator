#!/usr/bin/env python3
"""Auto-Build Pipeline CLI — PLAN → CODE → QA → SHIP

Usage:
    # Build from a GitHub issue
    python scripts/run_pipeline.py --owner Nietzsche-Ubermensch --repo goofy --issue-number 42

    # Build from a free-text requirement
    python scripts/run_pipeline.py --owner Nietzsche-Ubermensch --repo goofy \
        --requirements "Add a dark mode toggle to the sidebar" --title "Dark mode toggle"

    # Resume an interrupted run
    python scripts/run_pipeline.py --resume <run_id>

    # Don't wait for CI (just create the PR)
    python scripts/run_pipeline.py --owner Nietzsche-Ubermensch --repo goofy \
        --issue-number 42 --no-wait-for-ci

    # List recent pipeline runs
    python scripts/run_pipeline.py --list
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_build import Orchestrator, PipelineStatus, StateStore
from hermes_github import GitHubConfig


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)


def cmd_run(args: argparse.Namespace) -> int:
    """Run the pipeline from an issue or requirement."""
    config = GitHubConfig.from_env()

    orch = Orchestrator(
        config=config,
        wait_for_ci=not args.no_wait_for_ci,
        ci_timeout=args.ci_timeout,
        workspace=args.workspace,
    )

    try:
        if args.resume:
            state = orch.resume(args.resume)
        elif args.issue_number:
            state = orch.run_from_issue(
                owner=args.owner,
                repo=args.repo,
                issue_number=args.issue_number,
                wait_for_ci=not args.no_wait_for_ci,
            )
        elif args.requirements:
            state = orch.run_from_requirement(
                owner=args.owner,
                repo=args.repo,
                requirements=args.requirements,
                title=args.title or "Manual build request",
                wait_for_ci=not args.no_wait_for_ci,
            )
        else:
            print("Error: must specify --issue-number, --requirements, or --resume")
            return 1

        if state is None:
            print("Error: pipeline run not found")
            return 1

        # Print summary
        print("\n" + "=" * 60)
        print(f"Pipeline Run: {state.run_id}")
        print(f"Status:      {state.status.value}")
        print(f"Phase:       {state.current_phase}")
        print(f"Trigger:     {state.trigger}")
        print(f"Repo:        {state.repo_owner}/{state.repo_name}")

        if state.plan:
            print(f"Branch:      {state.plan.branch_name}")
            print(f"Commit:      {state.commit_sha[:7] if state.commit_sha else 'N/A'}")
            print(f"Files:       {len(state.files_committed)}")

        if state.qa_result:
            print(f"QA:          {'✅ Passed' if state.qa_result.passed else '❌ Failed'}")

        if state.ship_result:
            print(f"PR:          #{state.ship_result.pr_number}" if state.ship_result.pr_number else "PR: N/A")
            print(f"PR URL:      {state.ship_result.pr_url}" if state.ship_result.pr_url else "")
            print(f"Merged:      {'✅ Yes' if state.ship_result.merged else '❌ No'}")

        if state.errors:
            print(f"\nErrors:")
            for err in state.errors:
                print(f"  - {err}")

        print("=" * 60)

        return 0 if state.status == PipelineStatus.COMPLETED else 1

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    finally:
        orch.close()


def cmd_list(args: argparse.Namespace) -> int:
    """List recent pipeline runs."""
    store = StateStore()
    runs = store.list_runs(limit=args.limit)

    if not runs:
        print("No pipeline runs found.")
        return 0

    print(f"{'Run ID':<36} {'Status':<12} {'Repo':<30} {'Phase':<12} {'Updated'}")
    print("-" * 110)

    for run in runs:
        repo = f"{run.repo_owner}/{run.repo_name}" if run.repo_owner else "—"
        print(
            f"{run.run_id:<36} {run.status.value:<12} {repo:<30} "
            f"{run.current_phase:<12} {run.updated_at.strftime('%Y-%m-%d %H:%M')}"
        )

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show details of a specific pipeline run."""
    store = StateStore()
    state = store.load(args.run_id)

    if state is None:
        print(f"Run {args.run_id} not found.")
        return 1

    print(json.dumps(state.model_dump(mode="json"), indent=2, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-Build Pipeline: PLAN → CODE → QA → SHIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument("--owner", default="Nietzsche-Ubermensch", help="Repo owner")
    run_parser.add_argument("--repo", required=False, help="Repo name")
    run_parser.add_argument("--issue-number", type=int, help="GitHub issue number")
    run_parser.add_argument("--requirements", help="Free-text requirements")
    run_parser.add_argument("--title", help="Title for manual builds")
    run_parser.add_argument("--resume", help="Resume a run by ID")
    run_parser.add_argument("--no-wait-for-ci", action="store_true", help="Don't wait for CI")
    run_parser.add_argument("--ci-timeout", type=float, default=600.0, help="CI wait timeout (seconds)")
    run_parser.add_argument("--workspace", help="Workspace path for local QA tools")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    # list command
    list_parser = subparsers.add_parser("list", help="List recent pipeline runs")
    list_parser.add_argument("--limit", type=int, default=20, help="Number of runs to show")

    # show command
    show_parser = subparsers.add_parser("show", help="Show details of a pipeline run")
    show_parser.add_argument("run_id", help="Run ID to show")

    # Default to run if no command
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    # Also allow running without subcommand (for backward compat)
    if sys.argv[1] not in ("run", "list", "show", "-h", "--help"):
        # Insert "run" as the default command
        sys.argv.insert(1, "run")

    args = parser.parse_args()

    setup_logging(getattr(args, "verbose", False))

    if args.command == "list":
        return cmd_list(args)
    elif args.command == "show":
        return cmd_show(args)
    elif args.command == "run" or args.command is None:
        return cmd_run(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
