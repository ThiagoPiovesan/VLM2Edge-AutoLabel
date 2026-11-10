import pytest

from vlm2edge.eval.benchmark import compute_latency_stats


def test_compute_latency_stats_empty() -> None:
    stats: dict[str, float] = compute_latency_stats([])
    assert stats == {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}


def test_compute_latency_stats_values() -> None:
    # 20 values from 1.0 to 20.0 ms -> mean = 10.5 ms, fps = 1000 / 10.5
    lats: list[float] = [float(x) for x in range(20, 0, -1)]
    stats: dict[str, float] = compute_latency_stats(lats)
    assert stats["mean_ms"] == pytest.approx(10.5)
    assert stats["p50_ms"] == pytest.approx(10.5, abs=0.5)
    assert stats["p95_ms"] == pytest.approx(19.0, abs=0.5)
    assert stats["fps"] == pytest.approx(1000.0 / 10.5)
