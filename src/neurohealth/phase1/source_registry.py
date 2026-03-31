from __future__ import annotations

from typing import Iterable

from .types import AllowedUsage, SourceRecord
from .utils import read_json


APPROVED_USAGE = {
    AllowedUsage.REDISTRIBUTABLE,
    AllowedUsage.REDISTRIBUTABLE_WITH_ATTRIBUTION,
}


def load_source_registry(path: str) -> list[SourceRecord]:
    records = [SourceRecord(**item) for item in read_json(path)]
    return records


def approved_sources(records: Iterable[SourceRecord]) -> list[SourceRecord]:
    return [
        record
        for record in records
        if record.status == "approved" and record.allowed_usage in APPROVED_USAGE
    ]


def source_lookup(records: Iterable[SourceRecord]) -> dict[str, SourceRecord]:
    return {record.source_id: record for record in records}
