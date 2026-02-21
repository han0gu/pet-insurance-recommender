from langchain_core.documents import Document

chunk = Document(
    page_content=('- 된 비용(각종 할인 및 감면, 사후환급금액 등을 제외한 실수납액을 의미합니다. 이하\n'
 '- 「의료비」 라 합니다)을 제4항에 따라 보험가입금액을 한도로 보험수익자에게 반려묘\n'
 '- 의료비(치과및구강질환포함)(재가입형) 보험금(이하 「의료비보험금」 라 합니다)으로\n'
 '- 보상하여 드립니다. 단, 보험기간 중에 발생한 사고로 회사가 지급하는 연간 의료비보\n'
 '- 험금의 총 합계는 보험증권에 기재된 연간 총 보상한도액을 한도로 합니다.\n'
 '- ② 반려묘가 제1항의 사고로 치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터'),
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
