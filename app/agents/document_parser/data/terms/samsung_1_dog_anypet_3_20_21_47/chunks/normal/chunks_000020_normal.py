from langchain_core.documents import Document

chunk = Document(
    page_content=('. 목욕 비용(약용 및 처방샴푸 값 포함) 및 벼룩, 진드기, 모낭충의 제거 비용 10. 한방 및 한약(보상하는 상해 또는 질병의 치료를 '
 '위한 침술 및 물리치료는 제외), 온천요법, 산 소요법, 면역요법 등의 대체적 처치에 의한 치료를 위한 비용'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000020',
              'chunk_char_len': 137,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
