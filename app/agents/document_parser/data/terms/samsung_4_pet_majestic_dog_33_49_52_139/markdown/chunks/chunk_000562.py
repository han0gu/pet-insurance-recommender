from langchain_core.documents import Document

chunk = Document(
    page_content=('비용\n'
 '8. 건강식품, 보조식품, 보조치료제 및 Supplement 비용(치료를 목적으로 하는지 불\n'
 '문합니다)\n'
 '9. 목욕 비용(약용 및 처방샴푸 값 포함) 및 벼룩, 진드기, 모낭충의 제거 비용\n'
 '10. 한방 및 한약(보상하는 상해 또는 질병의 치료를 위한 침술 및 물리치료는 제외합\n'
 '니다), 온천요법, 산소요법, 면역요법 등의 대체적 처치에 의한 치료를 위한 비용\n'
 '11. 가입동물의 이송비, 마이크로칩의 삽입 비용, 안락사를 위한 비용, 장례식비용, 매\n'
 '장비용 등 가입동물의 사망 후에 소요된 비용, 각종 증명서류의 작성비용(운송비'),
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
