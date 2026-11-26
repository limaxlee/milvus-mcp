from datetime import datetime, timezone


def get_current_utc_time_as_string() -> str:
    return datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S.%fZ')
