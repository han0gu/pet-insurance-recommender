from langchain_core.documents import Document

chunk = Document(
    page_content=('- 체상해를 입은 때에만 보상합니다.\n'
 '- ③ 제1항 및 제2항에도 불구하고 제1항 제2호 내지 제4호에 해당하는 강력범죄에 의하여\n'
 '- 피보험자가 사망하였을 경우에는 제1항 제1호의 살인에 해당하는 것으로 봅니다.\n'
 '# 제3조 (보험금 지급에 관한 세부규정)- \n'
 '- 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못\n'
 '- 할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있습니다.\n'
 '- 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하며, 보험금 지'),
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
