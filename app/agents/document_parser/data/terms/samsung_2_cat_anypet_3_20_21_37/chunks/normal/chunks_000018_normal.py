from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 상병명을 알 수 없는 상해 또는 질병에 대한 치료 라. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약·예방 접종비용 및 '
 '정기검진, 예방적 검사를 위한 비용 마. 대상 반려동물의 정상적인 임신·출산, 제왕절개, 인공유산과 관련된 비용 및 출산 후 증상 치료 '
 '비용 바. 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용 사. 미용으로 인한 비용 아. 귀 성형, 꼬리 성형, 성대제거 '
 '및 미용성형을 위한 수술 및 처치에 따른 비용 자'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000018',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
