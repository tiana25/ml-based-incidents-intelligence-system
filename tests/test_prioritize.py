import pytest

from src.pipeline.prioritize import escalate_to_high, score_priority


def test_critical_outage_scores_high():
    result = score_priority("CRITICAL: auth service outage", "alert")
    assert result["priority"] == "high"


def test_retry_log_scores_medium():
    result = score_priority("Retry attempt 2 for DNS lookup", "log")
    assert result["priority"] == "medium"


def test_maintenance_ticket_scores_low():
    result = score_priority("Scheduled maintenance window", "ticket")
    assert result["priority"] == "low"


def test_score_is_in_unit_range():
    result = score_priority("CRITICAL: outage detected", "alert")
    assert 0.0 <= result["score"] <= 1.0


def test_alert_base_elevates_neutral_text_to_medium():
    result = score_priority("System check completed", "alert")
    assert result["priority"] == "medium"


def test_log_base_keeps_neutral_text_low():
    result = score_priority("System check completed", "log")
    assert result["priority"] == "low"


def test_high_keyword_overrides_low_source_base():
    result = score_priority("Service is down", "log")
    assert result["priority"] == "high"


def test_escalate_to_high_with_multiple_source_types():
    incidents = [
        {"id": 1, "source_type": "ticket", "priority": "low"},
        {"id": 2, "source_type": "log", "priority": "low"},
        {"id": 3, "source_type": "alert", "priority": "medium"},
    ]
    escalated = escalate_to_high(incidents)
    assert all(inc["priority"] == "high" for inc in escalated)


def test_escalate_preserves_other_fields():
    incidents = [
        {"id": 1, "source_type": "ticket", "priority": "low"},
        {"id": 2, "source_type": "log", "priority": "low"},
    ]
    escalated = escalate_to_high(incidents)
    assert escalated[0]["id"] == 1
    assert escalated[1]["id"] == 2


def test_no_escalation_for_single_source_type():
    incidents = [
        {"id": 1, "source_type": "log", "priority": "medium"},
        {"id": 2, "source_type": "log", "priority": "medium"},
    ]
    result = escalate_to_high(incidents)
    assert all(inc["priority"] == "medium" for inc in result)


def test_escalate_empty_list_returns_empty():
    assert escalate_to_high([]) == []
