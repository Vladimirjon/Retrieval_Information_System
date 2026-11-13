"""Métricas de recuperación: P@k, R@k, AP@k y MAP (convención TREC)."""


def _validate(ranking: list[str], k: int) -> None:
    if k <= 0:
        raise ValueError(f"k debe ser positivo, se recibió {k}")
    if len(set(ranking)) != len(ranking):
        raise ValueError("el ranking contiene doc_id duplicados")


def _hits_at_k(ranking: list[str], relevant: set[str], k: int) -> int:
    return sum(1 for doc_id in ranking[:k] if doc_id in relevant)


def precision_at_k(ranking: list[str], relevant: set[str], k: int = 10) -> float:
    _validate(ranking, k)
    # Se divide siempre por k: un ranking con menos de k documentos se
    # penaliza como si las posiciones faltantes fueran no relevantes.
    return _hits_at_k(ranking, set(relevant), k) / k


def recall_at_k(ranking: list[str], relevant: set[str], k: int = 10) -> float:
    _validate(ranking, k)
    relevant = set(relevant)
    if not relevant:
        return 0.0
    # Se divide por todos los relevantes de la consulta, no solo por los
    # recuperados en el top-k.
    return _hits_at_k(ranking, relevant, k) / len(relevant)


def average_precision(ranking: list[str], relevant: set[str], k: int = 1000) -> float:
    _validate(ranking, k)
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for position, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / position
    # Se divide por el total de relevantes, no por los encontrados antes del
    # corte: un relevante fuera del top-k aporta 0 pero sigue en el denominador.
    return precision_sum / len(relevant)


def mean_average_precision(
    run: dict[str, list[str]], qrels: dict[str, set[str]]
) -> float:
    if not qrels:
        return 0.0
    # Se promedia sobre todas las consultas de qrels: una consulta sin ranking
    # en el run aporta AP=0 en lugar de quedar fuera del denominador.
    total = sum(
        average_precision(run.get(qid, []), relevant) for qid, relevant in qrels.items()
    )
    return total / len(qrels)
