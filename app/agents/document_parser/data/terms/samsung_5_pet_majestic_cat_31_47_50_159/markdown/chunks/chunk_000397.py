from langchain_core.documents import Document

chunk = Document(
    page_content=('른 연명의료중단 등 결정 및 그 이행으로 피보험자가 사망하는 경우 연명의료중단 등\n'
 '결정 및 그 이행은 제1조(보험금의 지급사유) ‘사망’의 원인 및 ‘사망보험금’ 지\n'
 '급에 영향을 미치지 않습니다.③ 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지\n'
 '못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있\n'
 '습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하'),
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
