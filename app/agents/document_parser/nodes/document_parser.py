import argparse
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_upstage.document_parse import OutputFormat
from langchain_upstage import UpstageDocumentParseLoader

from rich import print as rprint

load_dotenv()


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run document parser graph.")
    parser.add_argument(
        "--file-name",
        help="PDF file name under app/agents/document_parser/data/terms",
    )
    return parser


def parse_document(
    file_name: str, output_format: OutputFormat = "html"
) -> List[Document]:
    # BASE_DIR = Path(os.getenv("PROJECT_ROOT", ".")).resolve() # project root
    BASE_DIR = Path(__file__).resolve().parent.parent  # app/agents/document_parser
    TERMS_DIR = BASE_DIR / "data" / "terms"
    TERM_FILE_PATH = TERMS_DIR / file_name
    rprint("🔗parse_document TERM_FILE_PATH:", TERM_FILE_PATH)

    dp_loader = UpstageDocumentParseLoader(
        file_path=str(TERM_FILE_PATH),
        output_format=output_format,
        coordinates=False,
    )
    # rprint(">>> dp_loader", dp_loader)

    rprint("🚀document parsing start. output format:", output_format)
    dp_result = dp_loader.load()  # 전체 문서가 하나의 Document로 추출됨
    rprint("✅document parsing done. result length:", len(dp_result))  # 1

    create_local_file(dp_result, TERMS_DIR, file_name, output_format)

    return split_by_page_marker(dp_result[0])


def split_by_page_marker(full_document: Document) -> list[Document]:
    """
    Upstage Document Parse 결과(전체가 1개 Document로 들어온 텍스트)를
    페이지 마커를 기준으로 페이지 단위 Document 리스트로 분리한다.

    기대하는 입력 예:
    - "... 약관 본문 ... - 1 - ... 다음 본문 ... - 2 - ..."

    반환:
    - 각 페이지를 `Document(page_content=..., metadata={"page": n, ...})` 형태로 반환
    - 원본 metadata는 유지하고, 페이지 번호(`page`)만 추가
    """

    # 전체 문서 텍스트(여러 페이지가 하나로 합쳐진 문자열)
    text = full_document.page_content

    # 페이지 마커 정규식:
    # - "- 21 -" 같은 형태를 인식한다.
    # - (\d+) 캡처 그룹으로 페이지 번호만 추출한다.
    #   예: "- 21 -" -> group(1) == "21"
    pattern = r"-\s*(\d+)\s*-"

    # 문서 전체에서 페이지 마커 위치를 모두 찾는다.
    # 각 match에는 "마커 시작/끝 인덱스"와 "페이지 번호 문자열"이 들어 있다.
    matches = list(re.finditer(pattern, text))
    rprint("📑total pages:", len(matches))

    # 마커를 하나도 찾지 못하면 페이지 분할이 불가능하므로 즉시 예외를 던진다.
    if not matches:
        raise ValueError("❗️페이지 마커를 찾지 못했습니다.")

    page_docs = []

    # 각 페이지 마커를 기준으로 "현재 마커 끝 ~ 다음 마커 시작" 구간을 잘라서
    # 페이지 본문으로 만든다.
    for i, match in enumerate(matches):
        # 정규식 캡처 그룹에서 페이지 번호를 꺼내 정수로 변환
        page_number = int(match.group(1))

        # start: 현재 페이지 마커 바로 뒤부터 본문 시작
        start = match.end()
        # end:
        # - 다음 마커가 있으면 그 직전까지
        # - 마지막 페이지면 문서 끝까지
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # 앞뒤 공백/개행 정리
        page_text = text[start:end].strip()

        # 페이지 단위 Document 생성
        # - 원본 metadata는 그대로 복사(**full_doc.metadata)
        # - page 번호를 추가해 downstream(청킹/태깅/인덱싱)에서 활용 가능하게 함
        page_docs.append(
            Document(
                page_content=page_text,
                metadata={
                    **full_document.metadata,
                    "page": page_number,
                },
            )
        )

    # 페이지 순서대로 분리된 Document 리스트 반환
    return page_docs


def create_local_file(
    dp_result: List[Document],
    target_dir: Path,
    original_file_name: str,
    output_format: OutputFormat = "html",
):
    if not dp_result:
        raise ValueError("Document parse result is empty.")

    output_extension_by_format = {
        "html": "html",
        "text": "txt",
        "markdown": "md",
    }
    OUTPUT_EXTENSION = output_extension_by_format.get(output_format)
    if OUTPUT_EXTENSION is None:
        raise ValueError(f"Unsupported output_format: {output_format}")
    rprint("OUTPUT_EXTENSION:", OUTPUT_EXTENSION)

    OUTPUT_FILE_NAME = f"{Path(original_file_name).stem}_dp_content.{OUTPUT_EXTENSION}"
    OUTPUT_FILE_PATH = target_dir / OUTPUT_FILE_NAME
    rprint("🔗create_local_file OUTPUT_FILE_PATH:", OUTPUT_FILE_PATH)

    if OUTPUT_FILE_PATH.exists():
        rprint("⚠️ create_local_file skipped (already exists)")
        return

    dp_content = dp_result[0].page_content
    # rprint(">>> dp_content", dp_content)

    rprint("🚀create_local_file start")
    OUTPUT_FILE_PATH.write_text(dp_content, encoding="utf-8")
    rprint("✅create_local_file done")


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    file_name = args.file_name
    parse_document(file_name)


# uv run python -m app.agents.document_parser.nodes.document_parse --file-name meritz_maum_pet_12_16.pdf
