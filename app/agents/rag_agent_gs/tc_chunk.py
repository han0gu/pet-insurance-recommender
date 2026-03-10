from pathlib import Path
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
PDF_FILES = {
    "chubb": BASE_DIR / "chubb.pdf",
    "kb": BASE_DIR / "KB.pdf",
    "meritz": BASE_DIR / "meritz.pdf",
}

DEFAULT_CORPUS = "meritz"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_pdf_as_pages(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """Load a PDF file page by page."""
    reader = PdfReader(str(pdf_path))
    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(
            {
                "page": i,
                "text": page.extract_text() or "",
            }
        )
    return pages


def pages_to_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    for page in pages:
        text = page["text"].strip()
        if not text:
            continue

        for chunk_index, chunk_text in enumerate(splitter.split_text(text)):
            chunks.append(
                {
                    "id": f"p{page['page']}_c{chunk_index}",
                    "page": page["page"],
                    "text": chunk_text,
                }
            )

    return chunks


def load_chunks(corpus: str = DEFAULT_CORPUS) -> List[Dict[str, Any]]:
    pdf_path = PDF_FILES[corpus]
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    return pages_to_chunks(load_pdf_as_pages(pdf_path))


def _load_default_chunks() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pdf_path = PDF_FILES[DEFAULT_CORPUS]
    if not pdf_path.exists():
        return [], []

    loaded_pages = load_pdf_as_pages(pdf_path)
    return loaded_pages, pages_to_chunks(loaded_pages)


pages, chunks = _load_default_chunks()


if __name__ == "__main__":
    print("pages:", len(pages))
    print("chunks:", len(chunks))
