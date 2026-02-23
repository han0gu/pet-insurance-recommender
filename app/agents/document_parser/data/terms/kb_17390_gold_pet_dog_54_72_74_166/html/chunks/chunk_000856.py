from langchain_core.documents import Document

chunk = Document(
    page_content=('회사가 제1회 보험료를 받고 승낙을 거절한 경우에는 거절통지와 함께 받은 금 상<br>액을 계약자에게 돌려 드리며, 보험료를 받은 기간에 '
 '대하여 평균공시이율 + 해<br>1%를 연단위 복리로 계산한 금액을 더하여 지급합니다'),
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
