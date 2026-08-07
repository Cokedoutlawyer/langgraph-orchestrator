"""Schema boundary validation and migration registry.

The boundary validator is the reusable mechanism that prevents unvalidated
data from crossing component boundaries. It validates a payload against the
expected canonical schema, captures errors as structured PipelineFailure
objects (never raw ValidationError), and provides a clean success/failure API.

The migration registry handles schema version evolution. When a payload's
schema_version differs from the current version, a registered migration
function is applied before validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from . import PipelineFailure, FailureCategory

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class BoundaryResult:
    """Result of a boundary validation."""
    success: bool
    validated: Any = None  # The validated model instance on success
    failure: PipelineFailure | None = None


class SchemaMigrations:
    """Centralized schema migration registry.

    Migrations are registered as (schema_name, from_version, to_version) →
    callable(dict) → dict. On validation, if the payload's schema_version
    differs from the target schema's current version, the chain of
    migrations is applied.
    """

    _migrations: dict[tuple[str, str, str], Any] = {}

    @classmethod
    def register(
        cls,
        schema_name: str,
        from_version: str,
        to_version: str,
        migration_fn: Any,
    ) -> None:
        key = (schema_name, from_version, to_version)
        cls._migrations[key] = migration_fn
        logger.debug("Migration registered: %s %s→%s", schema_name, from_version, to_version)

    @classmethod
    def migrate(cls, schema_name: str, payload: dict, target_version: str) -> dict:
        """Apply migrations to bring payload up to target_version."""
        current = payload.get("schema_version", "1.0")
        if current == target_version:
            return payload

        # Try to find a migration chain
        migrated = dict(payload)
        visited = set()
        max_steps = 10  # Prevent infinite loops

        while current != target_version and max_steps > 0:
            if current in visited:
                logger.warning("Migration cycle detected for %s at %s", schema_name, current)
                break
            visited.add(current)
            max_steps -= 1

            # Find a migration from current to any higher version
            found = False
            for (name, from_v, to_v), fn in cls._migrations.items():
                if name == schema_name and from_v == current:
                    migrated = fn(migrated)
                    migrated["schema_version"] = to_v
                    current = to_v
                    found = True
                    break

            if not found:
                logger.warning("No migration for %s from %s to %s", schema_name, current, target_version)
                break

        return migrated


class SchemaBoundary:
    """Reusable boundary validation mechanism.

    Usage:
        result = SchemaBoundary.validate(
            stage="qa",
            schema=QAReport,
            payload={"run_id": "...", "passed": True, ...},
        )
        if result.success:
            qa_report = result.validated
        else:
            failure = result.failure
    """

    @staticmethod
    def validate(
        stage: str,
        schema: Type[T],
        payload: Any,
        run_id: str = "",
    ) -> BoundaryResult:
        """Validate a payload against a canonical schema.

        Args:
            stage: Name of the pipeline stage (for failure attribution).
            schema: The Pydantic model class to validate against.
            payload: The raw payload (dict or model instance).
            run_id: Pipeline run ID for correlation.

        Returns:
            BoundaryResult with .success, .validated (on success), or
            .failure (on failure with structured PipelineFailure).
        """
        # If already a validated model instance of the right type, return it
        if isinstance(payload, schema):
            return BoundaryResult(success=True, validated=payload)

        # Convert to dict if needed
        if isinstance(payload, BaseModel):
            data = payload.model_dump()
        elif isinstance(payload, dict):
            data = payload
        else:
            return BoundaryResult(
                success=False,
                failure=PipelineFailure(
                    stage=stage,
                    run_id=run_id,
                    category=FailureCategory.SCHEMA_VALIDATION,
                    message=f"Expected dict or {schema.__name__}, got {type(payload).__name__}",
                ),
            )

        # Apply migrations if needed
        schema_name = schema.__name__
        # Get the default schema_version from the model
        try:
            default_instance = schema()
            target_version = default_instance.schema_version
        except Exception:
            target_version = "1.0"

        if "schema_version" in data and data["schema_version"] != target_version:
            data = SchemaMigrations.migrate(schema_name, data, target_version)

        # Validate
        try:
            validated = schema(**data)
            return BoundaryResult(success=True, validated=validated)
        except ValidationError as e:
            # Capture as structured failure — never expose raw ValidationError
            error_details = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "unknown error")
                error_details.append(f"{loc}: {msg}")

            failure = PipelineFailure(
                stage=stage,
                run_id=run_id,
                category=FailureCategory.SCHEMA_VALIDATION,
                retryable=False,
                message=f"Schema validation failed for {schema_name}",
                details="; ".join(error_details),
            )
            logger.warning("Boundary validation failed at %s: %s", stage, failure.details)
            return BoundaryResult(success=False, failure=failure)
        except Exception as e:
            failure = PipelineFailure(
                stage=stage,
                run_id=run_id,
                category=FailureCategory.SCHEMA_VALIDATION,
                retryable=False,
                message=f"Unexpected error validating {schema_name}: {e}",
            )
            return BoundaryResult(success=False, failure=failure)
