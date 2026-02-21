from langchain_core.documents import Document

chunk = Document(
    page_content=('수익자<br>가 보험금을 직접 청구할 수 없는 특별한 사정이 있음을 증명하는 서류를 제출하고 회<br>사의 승낙을 얻어 '
 '제1조(적용대상)의 수익자의 대리인으로서 보험금(사망보험금 제외)<br>을 청구하고 수령할 수 있습니다'),
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
