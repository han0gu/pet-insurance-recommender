from langchain_core.documents import Document

chunk = Document(
    page_content=('비용\n'
 '8. 입원중의 식이(食餌)에 해당하지 않는 음식물 및 식이요법, 그리고 수의사가 처방\n'
 '하는 의약품 이외의 것(건강보조식품, 의약품지정이 되어 있지 않은 한방약, 의약\n'
 '부외품 등)\n'
 '9. 목욕 비용(약욕 및 처방샴푸 값 포함) 및 귀 세정제(이어 클리너), 예방 가능한 기\n'
 '생충(벼룩, 진드기, 모낭충 등)의 제거 비용 및 기생충으로 발생한 질병의 치료비\n'
 '10. 한방 및 한약(보상하는 상해 또는 질병의 치료를 위한 침술 및 물리치료는 제외합\n'
 '니다), 온천요법, 산소요법, 면역요법 등의 대체적 처치에 의한 치료를 위한 비용'),
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
