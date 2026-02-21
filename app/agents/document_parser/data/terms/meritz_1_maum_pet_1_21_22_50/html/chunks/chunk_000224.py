from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험금 청구서(회사양식)<br>2. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인이 아닌<br>경우에는 '
 '본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보된 전자<br>적 수단을 활용한 피보험자 의사표시의 확인방법 '
 '포함)<br>3. 손해배상금 및 그 밖의 비용을 지급하였음을 명하는 서류<br>4'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
