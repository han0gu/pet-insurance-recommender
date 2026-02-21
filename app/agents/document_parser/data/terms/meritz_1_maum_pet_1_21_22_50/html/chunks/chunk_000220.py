from langchain_core.documents import Document

chunk = Document(
    page_content=('. 사고가 발생하였을 경우 사고가 발생한 때와 곳, 피해자의 주소와 성명, 사고상황<br>및 이들 사항의 증인이 있을 경우 그 주소와 '
 '성명<br>2. 피해자로부터 손해배상청구를 받았을 경우<br>3'),
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
