from industrial_ai.maintenance.contracts import SensorRecord


def validate_sensor_record(record: dict) -> list[str]:
    """Validate a sensor record against its declared data contract."""
    try:
        SensorRecord.model_validate(record)
    except Exception as exc:
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            return [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]
        raise

    return []
