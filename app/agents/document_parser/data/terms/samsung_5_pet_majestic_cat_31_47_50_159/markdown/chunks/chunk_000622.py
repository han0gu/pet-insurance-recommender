from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에서 수의사에게 치료 중자기공명영상(MRI) 또는 컴퓨터단층촬영(CT)을 시행한 경우\n'
 '- 연간 1회에 한하여 당일 피보험자가 부담한 반려묘의 치료에 사용된 비용(각종 할인\n'
 '- 및 감면, 사후환급금액 등을 제외한 실수납액을 의미합니다. 이하「의료비」 라 합니다\n'
 '- )을 제4항에 따라 보험가입금액을 한도로 보험수익자에게 보상하여 드립니다.\n'
 '- ② 반려묘가 제1항의 사고로 치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터\n'
 '- 180일 이내의 의료비는 보상하여 드립니다. 다만, 사고일 또는 발병일부터 365일이내'),
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
