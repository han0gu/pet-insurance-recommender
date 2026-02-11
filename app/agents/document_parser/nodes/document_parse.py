import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_upstage import UpstageDocumentParseLoader

from rich import print as rprint

load_dotenv()


def parse_document(file_name: str) -> List[Document]:
    PROJECT_ROOT_DIR = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
    TERMS_BASE_DIR = PROJECT_ROOT_DIR / "data" / "terms"
    FILE_PATH = TERMS_BASE_DIR / file_name
    print(">>> parse_document FILE_PATH\n", FILE_PATH)

    dp_loader = UpstageDocumentParseLoader(
        file_path=str(FILE_PATH),
        output_format="text",
        # output_format="markdown",
        # coordinates=False
    )
    rprint("dp_loader", dp_loader)

    dp_result = dp_loader.load()
    print("dp_result len", len(dp_result))  # 1
    # rprint(">>> sample dp_result", dp_result)

    return dp_result


# if __name__ == "__main__":
#     parse_document("meritz_terms_normal_1_5.pdf")
