from langchain_core.documents import Document

chunk = Document(
    page_content=('따라 계약자는 기존 계약에<br>이어 재가입할 수 있으며, 이 경우 회사는 기존계약의 가입 이후 발생한 상해 또<br>는 질병을 사유로 '
 '가입을 거절할 수 없습니다.<br>1'),
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
