from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 반려묘의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할 수 있는 증상을 포함합니다. 다만 보험기간 중 최초로 '
 '발견된 경우에는 보상합니다 .) 2. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약 · 예방 접종비용 및 정기검 진, 예방적 '
 '검사를 위한 비용 3. 임신, 출산(제왕절개를 포함합니다), 인공유산과 관련된 비용 및 출산 후 증상 치료 비용 4. 중성화, 불임 및 '
 '피임을 목적으로 한 수술 및 처치에 따른 비용 5'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 114},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000716',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
