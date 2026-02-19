from langchain_core.documents import Document

chunk = Document(
    page_content=('. 귀 성형, 꼬리 성형, 성대제거 및 미용성형을 위한 수술 및 처치에 따른 비용 자. 손톱절제(며느리발톱 제거 포함), 잔존유치, '
 '잠복고환, 배꼽허니아(배꼽부위탈장), 항문낭 제 거 등 건강동물에 실시하는 외과수술 및 기타 검사 또는 점안, 귀청소 등의 관리 비용'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['skin', 'other']},
 'indexing': {'chunk_id': 'chunk_000019',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
