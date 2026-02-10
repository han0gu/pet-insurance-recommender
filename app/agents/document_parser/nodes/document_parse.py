from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_upstage import UpstageDocumentParseLoader

from rich import print as rprint

load_dotenv()


def parse_document(file_name: str) -> List[Document]:
    BASE_DIR = Path(__file__).resolve().parent.parent  # app/agents/rag_agent/
    DEFAULT_FILE_PATH = BASE_DIR.joinpath("data", "terms", file_name)
    print("DEFAULT_FILE_PATH", DEFAULT_FILE_PATH)

    dp_loader = UpstageDocumentParseLoader(
        file_path=str(DEFAULT_FILE_PATH),
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
