from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료를 감<br>액하고, 이후 기간 보장을 위한 재원인 계약자적립액 등의 차이로 인하여 발생한<br>정산금액(이하 "정산금액"이라 '
 '합니다)을 환급하여 드립니다'),
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
