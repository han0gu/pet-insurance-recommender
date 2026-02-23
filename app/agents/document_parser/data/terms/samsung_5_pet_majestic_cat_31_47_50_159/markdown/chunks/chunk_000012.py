from langchain_core.documents import Document

chunk = Document(
    page_content=('- 화된 때에는 그 악화된 장해상태를 기준으로 장해지급률을 결정합니다.\n'
 '- ③ 장해분류표에 해당되지 않는 후유장해는 피보험자의 직업, 연령, 신분 또는 성별 등에\n'
 '- 관계없이 신체의 장해정도에 따라 장해분류표의 구분에 준하여 지급액을 결정합니다.\n'
 '- 다만, 장해분류표의 각 장해분류별 최저 지급률 장해정도에 이르지 않는 후유장해에\n'
 '- 대하여는 상해 후유장해(80%이상)보험금을 지급하지 않습니다.\n'
 '④ 보험수익자와 회사가 제3조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지'),
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
