from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다.<br>\uf000 제3항 및 제4항에도 불구하고, 회사가 계약자의 재가입 의사를 확인하지 못한 경<br>우(계약자와의 연락두절로 '
 '회사의 안내가 계약자에게 도달하지 못한 경우 포함)<br>에는 직전계약과 동일한 조건으로 보험계약을 연장합니다'),
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
