from langchain_core.documents import Document

chunk = Document(
    page_content='보<br>험수익자에게 지급합니다.<br>\uf000 제1항의 사망은 동물병원에서 적법하게 시행된 안락사를 포함합니다',
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
