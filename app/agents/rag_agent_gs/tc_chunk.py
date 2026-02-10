import os
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_FILENAME_1 = "chubb.pdf" 
PDF_PATH_1 = f"c:/Users/GS/Documents/workspace/pet-insurance-recommender/app/agents/rag_agent_gs/{PDF_FILENAME_1}"

PDF_FILENAME_2 = ".pdf" 
PDF_PATH_2 = f"c:/Users/GS/Documents/workspace/pet-insurance-recommender/app/agents/rag_agent_gs/{PDF_FILENAME_2}"

PDF_FILENAME_3 = ".pdf" 
PDF_PATH_3 = f"c:/Users/GS/Documents/workspace/pet-insurance-recommender/app/agents/rag_agent_gs/{PDF_FILENAME_3}"

# 파일 존재 여부 확인
assert os.path.exists(PDF_PATH_1), f"❌ PDF 파일을 찾을 수 없습니다: {PDF_PATH_1}"
# print("✅ PDF 파일 확인 완료:", PDF_PATH_1)

def load_pdf_as_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """PDF를 페이지 단위로 로드하여 각 페이지의 텍스트와 메타데이터(페이지 번호)를 추출합니다."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        pages.append({
            "page": i + 1,    # 페이지 번호 (1부터 시작)
            "text": text      # 페이지 텍스트
        })
    return pages
pages = load_pdf_as_pages(PDF_PATH_1)
# ================================================================
CHUNK_SIZE = 300 # 1) 적절한 청크 사이즈 설정하기
CHUNK_OVERLAP = 50 # 2) 적절한 청크 overlap 사이즈 설정하기
# ================================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)
def pages_to_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []

    for p in pages:
        if not p["text"].strip(): continue

        split_texts = splitter.split_text(p["text"])

        for j, t in enumerate(split_texts):
            chunks.append({
                "id": f"p{p['page']}_c{j}",
                "page": p["page"],
                "text": t
            })

    return chunks
chunks = pages_to_chunks(pages)  