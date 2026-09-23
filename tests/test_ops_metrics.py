from datetime import datetime, timedelta, timezone

import pytest

from peterbot.ops_metrics import MetricStore


def test_private_bounded_stage_summary_and_restart(tmp_path):
    path = tmp_path / "metrics.sqlite3"
    store = MetricStore(path)
    for value in range(1, 21):
        store.record("model", "ok" if value < 20 else "timeout", value * 100,
                     input_tokens=10, output_tokens=5)
    store.db.close()
    report = MetricStore(path).summary()["model"]
    assert report == {"count": 20, "p50_ms": 1000, "p95_ms": 1900,
                      "outcomes": {"ok": 19, "timeout": 1},
                      "input_tokens": 200, "output_tokens": 100}


def test_reject_unbounded_and_private_metric_fields(tmp_path):
    store = MetricStore(tmp_path / "metrics.sqlite3")
    with pytest.raises(ValueError):
        store.record("private_message", "ok", 100)
    with pytest.raises(ValueError):
        store.record("model", "raw_exception", 100)
    with pytest.raises(ValueError):
        store.record("model", "ok", 3_600_001)
    with pytest.raises(TypeError):
        store.record("model", "ok", 10, prompt="secret")
    assert store.summary() == {}


def test_retention_cutoff_requires_timezone(tmp_path):
    store = MetricStore(tmp_path / "metrics.sqlite3")
    store.record("queue", "ok", 2)
    with pytest.raises(ValueError):
        store.delete_before(datetime.now())
    assert store.delete_before(datetime.now(timezone.utc) + timedelta(days=1)) == 1
    assert store.summary() == {}
