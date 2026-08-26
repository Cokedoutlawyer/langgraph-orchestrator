"""Tests for schema boundary validation."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from pipeline_schemas.boundary import SchemaBoundary, BoundaryResult
from pipeline_schemas import (
    QAReport, CheckResult, SecurityResult,
    PipelineFailure, FailureCategory,
    PipelineResult, PipelineStatus,
    HermesRunRequest, TriggerType,
)


class TestSchemaBoundary:
    """Tests for the reusable boundary validation mechanism."""

    def test_valid_schema_continues(self):
        """Valid schema passes validation and returns the validated model."""
        payload = {
            "run_id": "test-1",
            "passed": True,
            "syntax_check": {"passed": True, "errors": []},
            "security_scan": {"passed": True, "findings": [], "critical_count": 0},
            "overall_message": "QA passed",
        }
        result = SchemaBoundary.validate("qa", QAReport, payload, run_id="test-1")
        assert result.success
        assert isinstance(result.validated, QAReport)
        assert result.validated.passed is True
        assert result.failure is None

    def test_invalid_schema_stops(self):
        """Invalid schema fails validation and returns structured failure."""
        payload = {
            "run_id": "test-2",
            # Missing 'passed' field which is required
            "syntax_check": {"passed": True, "errors": []},
        }
        result = SchemaBoundary.validate("qa", QAReport, payload, run_id="test-2")
        assert not result.success
        assert result.validated is None
        assert result.failure is not None
        assert result.failure.category == FailureCategory.SCHEMA_VALIDATION
        assert result.failure.stage == "qa"
        assert "passed" in result.failure.details  # Error mentions the missing field

    def test_raw_validation_error_not_exposed(self):
        """Pydantic ValidationError is captured, not raised as exception."""
        payload = {"run_id": "test-3", "passed": "not_a_bool"}  # Wrong type
        result = SchemaBoundary.validate("qa", QAReport, payload, run_id="test-3")
        assert not result.success
        assert result.failure is not None
        # The failure should be a PipelineFailure, not a ValidationError
        assert isinstance(result.failure, PipelineFailure)
        assert not isinstance(result.failure, Exception)

    def test_already_validated_model_passes_through(self):
        """If payload is already a validated model instance, it passes through."""
        qa = QAReport(run_id="test-4", passed=True)
        result = SchemaBoundary.validate("qa", QAReport, qa, run_id="test-4")
        assert result.success
        assert result.validated is qa

    def test_non_dict_payload_fails_gracefully(self):
        """Non-dict, non-model payload fails with structured error."""
        result = SchemaBoundary.validate("test", QAReport, "not_a_dict")
        assert not result.success
        assert result.failure is not None
        assert "Expected dict" in result.failure.message

    def test_boundary_failure_has_correlation(self):
        """Failure carries the run_id for correlation."""
        result = SchemaBoundary.validate("test", QAReport, {}, run_id="correlation-test")
        assert result.failure is not None
        assert result.failure.run_id == "correlation-test"
