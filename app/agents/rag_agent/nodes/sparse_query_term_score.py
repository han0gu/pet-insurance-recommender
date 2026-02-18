# sparse embedding model - simple version for inference only
from typing import List, Dict, Any
import math
from collections import defaultdict
import sys
import os
import argparse
import importlib.util
import json
from pathlib import Path
from kiwipiepy import Kiwi
from pypdf import PdfReader
from dotenv import load_dotenv

from app.agents.rag_agent.state.rag_state import RagState
from app.agents.rag_agent_gs.sparse import (
    calculate_tfidf_weights,
    match_predefined_words,
    tokenize_korean,
)

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(ENV_PATH)

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _get_terms_dir() -> Path:
    try:
        from app.agents.document_parser.constants import TERMS_DIR

        return Path(TERMS_DIR)
    except Exception:
        return (
            Path(__file__).resolve().parents[2] / "document_parser" / "data" / "terms"
        )


def _get_vocab_jsond_path_for_doc(file_name: str) -> Path | None:
    if not file_name:
        return None

    terms_dir = _get_terms_dir()
    doc_stem = Path(file_name).stem
    vocab_path = terms_dir / doc_stem / "vocab.jsond"
    if vocab_path.exists():
        return vocab_path

    return None


def _load_vocab_jsond(vocab_path: Path) -> Dict[str, Any]:
    try:
        with vocab_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def sparse_qt_score(state: RagState) -> Dict[str, Any]:
    """
    Extract insurance product information from retrieved documents.

    Processes the top 5 retrieved documents from RAG retrieval and extracts
    insurance product and insurer metadata for recommendation.

    Args:
        state: RagState containing retrieved_documents from vector DB search

    Returns:
        Updated state dict with:
        - "sparse_scoring_results": List of top 5 products with:
          {
            "product_name": str,
            "insurer": str,
            "metadata": dict
          }
    """

    vocab_cache: Dict[str, Dict[str, Any]] = {}

    for document in state.retrieved_documents:
        evaluation = document.metadata.setdefault("evaluation", {})
        query_text = evaluation.get("query_context", "")
        doc_meta = document.metadata.get("doc", {})
        file_name = doc_meta.get("file_name", "")
        vocab_path = _get_vocab_jsond_path_for_doc(file_name)

        if vocab_path is None:
            evaluation["sparse_score"] = 0.0
            continue

        query_tokens = tokenize_korean(query_text)
        vocab_key = str(vocab_path)
        if vocab_key not in vocab_cache:
            vocab_cache[vocab_key] = _load_vocab_jsond(vocab_path)

        vocab_data = vocab_cache[vocab_key]
        vocab = vocab_data.get("vocab", {})
        idf = vocab_data.get("idf")
        predefined_words = vocab_data.get("predefined_words", [])

        if not isinstance(idf, dict):
            idf = {token: 1.0 for token in vocab.keys()}

        matched_vocab_tokens = [t for t in query_tokens if t in vocab]
        unmatched_vocab_tokens = [t for t in query_tokens if t not in vocab]
        vocab_weights = calculate_tfidf_weights(matched_vocab_tokens, idf)
        vocab_total = sum(vocab_weights.values())

        matched_predefined = match_predefined_words(query_text, predefined_words)
        predefined_weights = calculate_tfidf_weights(matched_predefined, idf)
        predefined_total = sum(predefined_weights.values())

        evaluation["sparse_score"] = vocab_total + predefined_total
        evaluation["sparse_debug"] = {
            "query_text": query_text,
            "query_tokens": query_tokens,
            "vocab_path": str(vocab_path),
            "matched_vocab_tokens": matched_vocab_tokens,
            "unmatched_vocab_tokens": unmatched_vocab_tokens,
            "matched_predefined": matched_predefined,
            "vocab_total": vocab_total,
            "predefined_total": predefined_total,
        }

    if state.retrieved_documents:
        print("\n[Sparse score results - top 5 chunks]")
        first_query_printed = False
        for idx, document in enumerate(state.retrieved_documents[:5], start=1):
            evaluation = document.metadata.get("evaluation", {})
            debug_info = evaluation.get("sparse_debug", {})
            doc_meta = document.metadata.get("doc", {})
            indexing_meta = document.metadata.get("indexing", {})
            file_name = doc_meta.get("file_name", "")
            page = doc_meta.get("page", "")
            chunk_id = doc_meta.get("chunk_id") or indexing_meta.get("chunk_id") or ""

            if not first_query_printed:
                print(f"query: {debug_info.get('query_text', '')}")
                first_query_printed = True

            print(f"\n{idx}. file={file_name} page={page} chunk_id={chunk_id}")
            chunk_content = " ".join(document.page_content.split())
            print(f"  chunk_content: {chunk_content}")

            if isinstance(debug_info, dict):
                query_tokens = debug_info.get("query_tokens", [])
                matched_vocab = debug_info.get("matched_vocab_tokens", [])
                unmatched_vocab = debug_info.get("unmatched_vocab_tokens", [])
                query_tokens_str = ", ".join(query_tokens)
                matched_vocab_str = ", ".join(matched_vocab)
                unmatched_vocab_str = ", ".join(unmatched_vocab)
                print(f"  query_tokens: [{query_tokens_str}]")
                print(f"  matched_vocab_tokens: [{matched_vocab_str}]")
                print(f"  unmatched_vocab_tokens: [{unmatched_vocab_str}]")

            total_score = evaluation.get("total_score", "N/A")
            sparse_score = evaluation.get("sparse_score", "N/A")
            print(f"  evaluation: ({total_score}, {sparse_score})")

    return {"retrieved_documents": state.retrieved_documents}
