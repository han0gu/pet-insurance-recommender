from langchain_core.documents import Document

chunk = Document(
    page_content=('말하며, 이하 "보험계약의 보험기간 전체"라 합니다)<br>로 하며, 그 판단기준은 회사에서 정한 계약사정기준(계약인수지침 등)을 '
 '따릅니<br>다'),
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
