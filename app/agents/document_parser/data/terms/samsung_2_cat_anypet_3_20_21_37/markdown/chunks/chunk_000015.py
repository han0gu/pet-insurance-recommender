from langchain_core.documents import Document

chunk = Document(
    page_content=('고양이범백혈구감소증, 고양이칼리시바이러스감염증, 고양이바이러스성비기관지염, 고양이백혈병\n'
 '바이러스감염증- 다. 상병명을 알 수 없는 상해 또는 질병에 대한 치료\n'
 '- 라. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약·예방 접종비용 및 정기검진, 예방적\n'
 '- 검사를 위한 비용\n'
 '- 마. 대상 반려동물의 정상적인 임신·출산, 제왕절개, 인공유산과 관련된 비용 및 출산 후 증상\n'
 '- 치료 비용\n'
 '- 바. 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용\n'
 '- 사. 미용으로 인한 비용'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
