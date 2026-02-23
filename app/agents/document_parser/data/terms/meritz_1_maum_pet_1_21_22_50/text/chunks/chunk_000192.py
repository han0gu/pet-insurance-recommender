from langchain_core.documents import Document

chunk = Document(
    page_content=('3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인이 아\n'
 '닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보된\n'
 '전자적 수단을 활용한 피보험자 의사표시의 확인방법 포함)② 회사는 제1항의 지정대리청구인 변경 지정시 계약자의 지정 편의를 위해 '
 '가족관계서류\n'
 '의 수령을 생략할 수 있습니다.- 42 -제5조(보험금 지급 등의 절차)① 지정대리청구인은 제6조(보험금의 청구)에 정한 구비서류 및 '
 '제1조(적용대상)의 수익자'),
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
