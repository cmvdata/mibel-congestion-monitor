"""
Une PDFs de una carpeta en un solo .txt con headers separadores claros,
listo para subir a NotebookLM como una sola source.

Uso:
    python scripts/merge_pdfs_to_txt.py <input_dir> <output_txt> [--pattern "*.pdf"]

Ejemplos:
    python scripts/merge_pdfs_to_txt.py \
        "C:/path/MIBEL_corpus/01_CNMC" \
        "C:/path/MIBEL_corpus/merged/CNMC_boletines_indicadores_2022.txt" \
        --pattern "CNMC_boletin_indicadores_2022_*.pdf"

Requiere: pip install pymupdf
"""

import argparse
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: falta PyMuPDF. Instala con:  pip install pymupdf", file=sys.stderr)
    sys.exit(1)


SEPARATOR = "=" * 78


def extract_pdf_text(pdf_path: Path) -> str:
    parts = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                parts.append(f"\n--- {pdf_path.name} | page {page_num} ---\n")
                parts.append(text)
    return "".join(parts)


def merge(input_dir: Path, output_txt: Path, pattern: str) -> None:
    pdfs = sorted(input_dir.glob(pattern))
    if not pdfs:
        print(f"ERROR: no se encontraron PDFs en {input_dir} con patrón {pattern}", file=sys.stderr)
        sys.exit(2)

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    total_words = 0
    with output_txt.open("w", encoding="utf-8") as out:
        out.write(f"{SEPARATOR}\n")
        out.write(f"CORPUS MERGE — {output_txt.stem}\n")
        out.write(f"Source folder: {input_dir}\n")
        out.write(f"Pattern: {pattern}\n")
        out.write(f"Files merged: {len(pdfs)}\n")
        out.write(f"{SEPARATOR}\n\n")

        for i, pdf in enumerate(pdfs, start=1):
            print(f"[{i}/{len(pdfs)}] {pdf.name}")
            out.write(f"\n{SEPARATOR}\n")
            out.write(f"DOCUMENT {i}: {pdf.name}\n")
            out.write(f"{SEPARATOR}\n")
            try:
                text = extract_pdf_text(pdf)
                out.write(text)
                total_words += len(text.split())
            except Exception as e:
                msg = f"[ERROR extrayendo {pdf.name}: {e}]"
                out.write(f"\n{msg}\n")
                print(f"  WARNING: {msg}", file=sys.stderr)

    size_mb = output_txt.stat().st_size / (1024 * 1024)
    print(f"\nOK -> {output_txt}")
    print(f"   files merged: {len(pdfs)}")
    print(f"   words: ~{total_words:,}")
    print(f"   size: {size_mb:.2f} MB")
    if total_words > 500_000:
        print(f"   WARNING: supera 500k palabras (limite NotebookLM). Divide por subgrupo.")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge PDFs to a single .txt for NotebookLM")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_txt", type=Path)
    p.add_argument("--pattern", default="*.pdf", help="Glob pattern (default: *.pdf)")
    args = p.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: {args.input_dir} no es un directorio", file=sys.stderr)
        sys.exit(1)

    merge(args.input_dir, args.output_txt, args.pattern)


if __name__ == "__main__":
    main()
