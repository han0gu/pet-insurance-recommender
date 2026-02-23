from langchain_core.documents import Document

chunk = Document(
    page_content='정신행동장해는 보험기간중에 발생한 뇌의 질병 또는 상해를 입은<br>후 18개월이 지난 후에 판정함을 원칙으로 한다',
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
