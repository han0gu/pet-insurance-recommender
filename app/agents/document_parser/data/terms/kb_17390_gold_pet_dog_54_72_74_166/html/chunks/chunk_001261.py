from langchain_core.documents import Document

chunk = Document(
    page_content=('. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인<br>이 아닌 경우에는 본인의 인감증명서, '
 '본인서명사실확인서 또는 안전성과 신<br>뢰성이 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)<br>6. 수탁기관 '
 '위탁비용 영수증 및 동물관리위탁업자가 제공하는 계약서(위탁관리<br>업소 등록번호, 업소명 및 주소, 전화번호, 위탁관리동물 종류, '
 '품종, 나이,<br>서비스 기간, 비용 등 포함)<br>7'),
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
