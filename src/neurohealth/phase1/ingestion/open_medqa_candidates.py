from __future__ import annotations

import logging
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


CANDIDATES = [
    {
        "candidate_id": "medquad",
        "name": "MedQuAD",
        "type": "medical_qa",
        "repo_api": "https://api.github.com/repos/abachaa/MedQuAD",
        "dataset_url": "https://github.com/abachaa/MedQuAD",
        "license_hint_url": "https://raw.githubusercontent.com/abachaa/MedQuAD/master/readme.txt",
        "license_hint_contains": "Creative Commons Attribution 4.0 International Licence (CC BY)",
    },
    {
        "candidate_id": "synthea",
        "name": "Synthea",
        "type": "synthetic_health_records",
        "repo_api": "https://api.github.com/repos/synthetichealth/synthea",
        "dataset_url": "https://github.com/synthetichealth/synthea",
    },
]


ALLOWED_LICENSE_KEYS = {
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
}


def evaluate_open_medqa_candidates(timeout: int = 20) -> dict[str, list[dict[str, Any]]]:
    admitted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []

    for item in CANDIDATES:
        payload = None
        try:
            resp = requests.get(item["repo_api"], timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read candidate metadata for %s: %s", item["candidate_id"], exc)
            quarantined.append(
                {
                    **item,
                    "status": "quarantine",
                    "reason": "metadata_unavailable",
                    "license_key": None,
                    "license_name": None,
                }
            )
            continue

        license_info = payload.get("license") or {}
        license_key = (license_info.get("key") or "").lower()
        license_name = license_info.get("name")
        dataset_type = item.get("type", "unknown")

        if license_key not in ALLOWED_LICENSE_KEYS and item.get("license_hint_url"):
            try:
                hint_resp = requests.get(item["license_hint_url"], timeout=timeout)
                hint_resp.raise_for_status()
                hint_text = hint_resp.text
                needle = item.get("license_hint_contains", "")
                if needle and needle in hint_text:
                    license_key = "cc-by-4.0"
                    license_name = "Creative Commons Attribution 4.0 International"
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to inspect license hint for %s: %s", item["candidate_id"], exc)

        if license_key not in ALLOWED_LICENSE_KEYS:
            quarantined.append(
                {
                    **item,
                    "status": "quarantine",
                    "reason": "license_not_redistributable_or_unclear",
                    "license_key": license_key or None,
                    "license_name": license_name,
                    "spdx": license_info.get("spdx_id"),
                }
            )
            continue

        if dataset_type != "medical_qa":
            quarantined.append(
                {
                    **item,
                    "status": "quarantine",
                    "reason": "dataset_type_not_medical_qa",
                    "license_key": license_key,
                    "license_name": license_name,
                    "spdx": license_info.get("spdx_id"),
                }
            )
            continue

        admitted.append(
            {
                **item,
                "status": "admitted",
                "license_key": license_key,
                "license_name": license_name,
                "spdx": license_info.get("spdx_id"),
            }
        )

    return {"admitted": admitted, "quarantined": quarantined}
