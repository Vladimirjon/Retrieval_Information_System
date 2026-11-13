"""Construye y comprueba artefactos clásicos o densos del sistema."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from ris.data import load_corpus
from ris.index import InvertedIndex


def _show_build_progress(processed: int, total: int, vocabulary_size: int) -> None:
    """Muestra el avance sin saturar la terminal."""
    if processed != total and processed % 1_000 != 0:
        return

    elapsed = perf_counter() - _show_build_progress.start_time
    rate = processed / elapsed if elapsed else 0.0
    percentage = processed / total * 100 if total else 100.0
    print(
        f"  Documentos: {processed:,}/{total:,} ({percentage:5.1f}%) | "
        f"vocabulario: {vocabulary_size:,} términos | "
        f"velocidad: {rate:,.0f} doc/s | transcurrido: {elapsed:.1f} s",
        flush=True,
    )


def _positive_argument(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Debe ser un entero positivo")
    return number


def _peak_memory_mib() -> float:
    """Pico residente del proceso, incluidas las asignaciones nativas del modelo."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD
        ]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / (1024 * 1024)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _build_dense(args: argparse.Namespace) -> None:
    from ris.retrievers.dense import DenseRetriever

    started = perf_counter()
    dirname = f"dense_sample_{args.limit}" if args.limit is not None else "dense"
    output = args.output_dir or Path("artifacts") / dirname
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError(
            f"{output} ya contiene archivos; use otra salida o --overwrite para reemplazar"
        )
    corpus = load_corpus()
    total_corpus = len(corpus)
    if args.limit is not None:
        corpus = {id_: corpus[id_] for id_ in sorted(corpus)[:args.limit]}
    print(f"Corpus: {len(corpus):,}/{total_corpus:,}; salida: {output}", flush=True)
    print(f"Batch: {args.batch_size}; hilos CPU: {args.threads}", flush=True)
    build_start = perf_counter()
    last_report = build_start

    def progress(processed: int, total: int) -> None:
        nonlocal last_report
        now = perf_counter()
        if now - last_report < 5 and processed != total:
            return
        elapsed = now - build_start
        rate = processed / elapsed
        print(
            f"  Documentos: {processed:,}/{total:,} ({processed / total * 100:.1f}%) | "
            f"{rate:.1f} doc/s | {elapsed:.1f} s | "
            f"restante estimado: {(total - processed) / rate:.1f} s",
            flush=True,
        )
        last_report = now

    retriever = DenseRetriever.build(
        corpus, batch_size=args.batch_size, threads=args.threads,
        progress_callback=progress,
    )
    build_seconds = perf_counter() - build_start
    retriever.save(output)
    loaded = DenseRetriever.load(
        output, encoder=retriever._encoder,
        expected_corpus_sha256=retriever.metadata["corpus_sha256"],
    )
    if loaded.doc_ids != retriever.doc_ids:
        raise ValueError("El mapeo recargado no coincide con el original")
    # Compartir el encoder evita cargar una segunda copia del mismo modelo CPU.
    query = "How should I invest my savings?"
    query_started = perf_counter()
    before = retriever.search(query)
    after = loaded.search(query)
    query_seconds = perf_counter() - query_started
    if before != after:
        raise ValueError("El ranking o los scores cambiaron al guardar y recargar")
    report = {
        "documents": len(corpus), "full_corpus_documents": total_corpus,
        "sample_limit": args.limit, "batch_size": args.batch_size,
        "threads": args.threads, "build_seconds": build_seconds,
        "verification_query": query, "query_verification_seconds": query_seconds,
        "verification_ranking": [{"doc_id": result.doc_id, "score": result.score}
                                 for result in after],
        "total_seconds": perf_counter() - started,
        "peak_process_memory_mib": _peak_memory_mib(),
        "memory_measure": "peak working set" if sys.platform == "win32" else "peak RSS",
        "artifact_bytes": {name: (output / name).stat().st_size for name in (
            "index.faiss", "embeddings.npy", "metadata.json"
        )},
        "corpus_sha256": retriever.metadata["corpus_sha256"],
        "model_revision": retriever.metadata["model_revision"],
        "max_seq_length": retriever.metadata["max_seq_length"],
    }
    with (output / "build_report.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Construcción y recarga verificadas: {build_seconds:.2f} s", flush=True)
    print(f"Pico de memoria del proceso: {report['peak_process_memory_mib']:.2f} MiB",
          flush=True)
    print(f"Artefactos: {sum(report['artifact_bytes'].values()) / 1024**2:.2f} MiB",
          flush=True)
    print(f"Reporte: {output / 'build_report.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye y comprueba el índice invertido o denso de FiQA."
    )
    parser.add_argument(
        "--stem",
        action="store_true",
        help="Aplica stemming de Porter al construir el índice.",
    )
    parser.add_argument("--dense", action="store_true", help="Construye MiniLM + FAISS.")
    parser.add_argument("--batch-size", type=_positive_argument, default=32)
    parser.add_argument("--threads", type=_positive_argument, default=8)
    parser.add_argument("--limit", type=_positive_argument,
                        help="Solo denso: muestra inicial ordenada por doc_id.")
    parser.add_argument("--output-dir", type=Path, help="Directorio de salida opcional.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Autoriza reemplazar artefactos densos existentes.")
    args = parser.parse_args()
    if args.dense and args.stem:
        parser.error("El denso usa texto original; --stem solo corresponde al clásico")
    if args.limit is not None and not args.dense:
        parser.error("--limit requiere --dense")
    if args.dense:
        _build_dense(args)
        return

    output_dir = args.output_dir or Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "index_stem.json" if args.stem else "index_no_stem.json"
    output_path = output_dir / filename

    print("Fase 1/5 - Cargando el corpus FiQA...", flush=True)
    corpus = load_corpus()
    print(f"  Corpus cargado: {len(corpus):,} documentos.", flush=True)

    print(
        f"Fase 2/5 - Construyendo el índice "
        f"({'con' if args.stem else 'sin'} stemming)...",
        flush=True,
    )
    start = perf_counter()
    _show_build_progress.start_time = start
    index = InvertedIndex.build(
        corpus,
        stem=args.stem,
        progress_callback=_show_build_progress,
    )
    elapsed = perf_counter() - start

    print("Fase 3/5 - Guardando el índice (se reemplaza el archivo anterior)...", flush=True)
    index.save(output_path)

    print("Fase 4/5 - Recargando el índice para comprobarlo...", flush=True)
    loaded = InvertedIndex.load(output_path, expected_stem=args.stem)

    if (
        loaded.N != index.N
        or loaded.postings != index.postings
        or loaded.df != index.df
        or loaded.doc_lengths != index.doc_lengths
    ):
        raise ValueError("El índice recargado no coincide con el original")

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print("Fase 5/5 - Construcción finalizada y verificada.", flush=True)
    print(f"  Documentos indexados: {index.N:,}", flush=True)
    print(f"  Términos en el vocabulario: {len(index.postings):,}", flush=True)
    print(f"  Stemming: {'sí' if index.stem else 'no'}", flush=True)
    print(f"  Tiempo de construcción: {elapsed:.2f} segundos", flush=True)
    print(f"  Tamaño del archivo: {size_mb:.2f} MiB", flush=True)
    print(f"  Índice guardado y verificado en: {output_path}", flush=True)


if __name__ == "__main__":
    main()
