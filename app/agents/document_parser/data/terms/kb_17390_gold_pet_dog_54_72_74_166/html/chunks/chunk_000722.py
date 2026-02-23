from langchain_core.documents import Document

chunk = Document(
    page_content='입원일당이 지급된\n최초입원일 보장재개\n최종입원일\n퇴원없이\n…\n계속 입원\n보장 보장제외 보장',
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
