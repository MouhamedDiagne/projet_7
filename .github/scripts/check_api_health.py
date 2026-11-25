"""Wait for the deployed API to finish loading its configured MLflow model."""

import json
import os
import time
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    """Require a healthy light release, optionally pinned to an expected run."""
    url = os.environ['HEALTH_URL']
    expected_run_id = os.environ.get('EXPECTED_RUN_ID', '').strip()
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=10) as response:
                payload = json.load(response)
            run_id = payload.get('model_run_id')
            if (payload.get('status') == 'ok' and payload.get('model_mode') == 'light'
                    and run_id and (not expected_run_id or run_id == expected_run_id)):
                print(f"Healthy light release: run {run_id}")
                return
            print('Waiting for expected healthy light release:', payload, flush=True)
        except (URLError, TimeoutError, OSError, ValueError) as error:
            print(f'Waiting for API startup: {type(error).__name__}: {error}', flush=True)
        time.sleep(10)
    raise SystemExit('API did not report the expected healthy release within 10 minutes.')


if __name__ == '__main__':
    main()
