from langchain_core.documents import Document

chunk = Document(
    page_content=('- .)\n'
 '- 11. 대한민국 이외 지역에서 발생한 사고 및 손해\n'
 '- 12. 수의사 자격이 없는 자의 치료행위로 인한 손해(수의사의 소견 및 처방에 의한 경\n'
 '- 우도 동일) 및 그로 인하여 가중된 손해\n'
 '② 회사는 아래의 의료비 및 비용 또는 손해는 보상하지 않습니다.\n'
 '1. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약 · 예방 접종비용 및 정기검- 100 -# 진, 예방적 검사를 위한 비용- '
 '2. 임신, 출산(제왕절개를 포함합니다.), 인공유산과 관련된 비용 및 출산 후 증상 치\n'
 '- 료 비용'),
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
