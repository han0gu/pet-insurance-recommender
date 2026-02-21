from langchain_core.documents import Document

chunk = Document(
    page_content=('- 만 지급합니다.\n'
 '- ② 반려묘가 제1항의 사고로 치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터\n'
 '- 180일 이내의 의료비는 보상하여 드립니다. 다만, 사고일 또는 발병일부터 365일이내\n'
 '- 의 치료인 경우에 한합니다.\n'
 '- ③ 회사가 지급할 제1항에서 정한 보험금은 피보험자가 부담한 이물제거 치료 당일 발생\n'
 '- 한 의료비에서 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관 및 4-2.\n'
 '- 반려묘 수술비(치과및구강질환포함) 확대보장(재가입형) 추가특별약관 지급보험금 합'),
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
