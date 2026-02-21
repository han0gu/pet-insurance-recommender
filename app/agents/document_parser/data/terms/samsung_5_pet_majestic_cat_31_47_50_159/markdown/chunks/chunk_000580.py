from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 등록묘의 경우에는 동물등록증 또는 등록번호\n'
 '- 3. 미등록묘의 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를 회사에 제\n'
 '- 출하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다.\n'
 '- 4. 사망을 확인할 수 있는 서류(동물폐사확인서, 동물화장증명서 등)\n'
 '- 5. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이\n'
 '- 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이\n'
 '- 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
