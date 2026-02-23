from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 미등록견의 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를 회사에 제출\n'
 '- 하시고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다.\n'
 '- 4. 사고증명서(진료비 내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 포함)\n'
 '- 및 의료비 영수증 등)\n'
 '- 5. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이\n'
 '- 아닌 경우에는 본인의 인감증명서 또는 안전성과 신뢰성이 확보된 전자적 수단을\n'
 '- 활용한 보험수익자 의사표시의 확인방법 포함)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
