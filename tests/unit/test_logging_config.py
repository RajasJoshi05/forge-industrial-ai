import json
import logging

from industrial_ai.common.logging_config import JsonFormatter, configure_logging


def test_json_formatter_outputs_structured_record():
    record = logging.LogRecord(
        name="industrial_ai.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=10,
        msg="Sensor validation failed",
        args=(),
        exc_info=None,
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "WARNING"
    assert payload["logger"] == "industrial_ai.test"
    assert payload["message"] == "Sensor validation failed"
    assert "timestamp" in payload


def test_logging_configuration_sets_requested_level():
    configure_logging(logging.WARNING)

    assert logging.getLogger().level == logging.WARNING
