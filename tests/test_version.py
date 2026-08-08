"""Tests for the pipeline_schemas package version constant."""

from pipeline_schemas import VERSION


def test_version():
    """The package should expose VERSION equal to the current release."""
    assert VERSION == "2.0.0"


def test_version_is_string():
    """VERSION should be a string so it can be consumed programmatically."""
    assert isinstance(VERSION, str)
