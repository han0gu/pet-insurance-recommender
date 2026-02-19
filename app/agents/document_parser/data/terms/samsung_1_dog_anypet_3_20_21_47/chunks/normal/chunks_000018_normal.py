from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약 · 예방 접종비용 및 정기검진, 예방적 검사를 위한 비용 2. 임신 · '
 '출산, 제왕절개, 인공유산과 관련된 비용 및 출산 후 증상 치료 비용 3. 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용 4. '
 '손톱의 절제(며느리발톱의 제거 포함), 잔존유치, 잠복고환, 배꼽허니아(배꼽부위탈장), 항문낭 제거 등 건강동물에 실시하는 외과수술 및 '
 '기타 검사 또는 손톱깎기 등의 처치비용 5. 미용으로 인한 비용 6'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000018',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
