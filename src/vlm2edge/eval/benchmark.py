def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """
    Calcular estatísticas de latência (em milissegundos) e throughput (FPS)
    a partir de uma lista de tempos de inferência `latencies_ms`.
    """
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}

    sorted_lats: list[float] = sorted(latencies_ms)
    n: int = len(sorted_lats)

    mean_ms: float = sum(sorted_lats) / n

    mid: int = n // 2
    p50_ms: float = (
        sorted_lats[mid] if n % 2 == 1 else (sorted_lats[mid - 1] + sorted_lats[mid]) / 2.0
    )

    p95_idx: int = int(0.95 * (n - 1))
    p95_ms: float = sorted_lats[p95_idx]

    fps: float = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }

