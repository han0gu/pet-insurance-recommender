from langchain_core.documents import Document

chunk = Document(
    page_content=('사유의 발생을 안 때에는 지체없이 그 사실을 회사에 알려야 합니다.제 7조 (보험금의 청구)# ① 보험수익자는 다음의 서류를 제출하고 '
 '보험금을 청구하여야 합니다.- 1. 청구서(회사양식)\n'
 '- 2. 사고증명서(진단서, 진료비계산서, 사망진단서, 장해진단서, 입원치료확인서, 의사\n'
 '- 처방전(처방조제비) 등)\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본인이\n'
 '- 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이'),
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
