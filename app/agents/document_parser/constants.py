from typing import Final, Literal

from app.agents.utils import get_parent_path

BASE_DIR = get_parent_path(__file__)  # app/agents/document_parser
TERMS_DIR = BASE_DIR / "data" / "terms"

TAG_TYPE = Literal["normal", "simple"]

VECTOR_STORE_NAME_BY_TAG_TYPE: Final[dict[TAG_TYPE, str]] = {
    "normal": "terms_normal_tag_dense",
    "simple": "terms_simple_tag_dense",
}
