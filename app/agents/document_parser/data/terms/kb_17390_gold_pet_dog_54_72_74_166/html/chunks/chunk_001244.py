from langchain_core.documents import Document

chunk = Document(
    page_content='. 성<br>\uf000 제1항에 의해 자동갱신을 적용할 경우 보험증권에 그 내용을 기재하여 드립니다',
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
