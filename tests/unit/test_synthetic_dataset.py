import pytest

from scripts.generate_synthetic_dataset import (
    is_failure_within_warning_window,
)


def test_failure_exactly_30_minutes_ahead_is_positive() -> None:
    assert is_failure_within_warning_window(100, [106]) is True


def test_failure_35_minutes_ahead_is_negative() -> None:
    assert is_failure_within_warning_window(100, [107]) is False


def test_failure_at_current_timestamp_is_negative() -> None:
    assert is_failure_within_warning_window(100, [100]) is False


def test_failure_already_in_the_past_is_negative() -> None:
    assert is_failure_within_warning_window(100, [95]) is False


def test_no_failure_events_produces_negative_label() -> None:
    assert is_failure_within_warning_window(100, []) is False


def test_failure_25_minutes_ahead_is_positive() -> None:
    assert is_failure_within_warning_window(100, [105]) is True


def test_invalid_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="interval_minutes"):
        is_failure_within_warning_window(
            observation_index=100,
            failure_indices=[106],
            interval_minutes=0,
        )


def test_negative_warning_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="warning_window_minutes"):
        is_failure_within_warning_window(
            observation_index=100,
            failure_indices=[106],
            warning_window_minutes=-5,
        )
