"""Construye los artefactos clásicos del sistema."""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye y comprueba el índice invertido de FiQA."
    )
    parser.add_argument(
        "--stem",
        action="store_true",
        help="Aplica stemming de Porter al construir el índice.",
    )
    args = parser.parse_args()

    output_dir = Path("artifacts")
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