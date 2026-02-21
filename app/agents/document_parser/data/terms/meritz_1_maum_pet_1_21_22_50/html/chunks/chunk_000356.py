from langchain_core.documents import Document

chunk = Document(
    page_content=('. 청구서(회사양식)<br>2. 사고증명서<br>3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증)<br>4. '
 '피보험자 및 지정대리청구인의 가족관계등록부(가족관계증명서) 및 주민등록등본<br>5'),
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
