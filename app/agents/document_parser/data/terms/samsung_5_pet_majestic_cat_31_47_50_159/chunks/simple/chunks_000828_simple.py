from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 유사계약 청약일 이후 제1항 제1호 또는 제2호에서 정 한 질병과 관련한 새로운 위험(재진단·치료 등은 해당하지 않습니다)이 '
 '발생하거나, 새로운 질병에 대한 보장이 추가(입원비, 수술비, 진단비 등 보장 범위의 변경 또는 확 대는 해당하지 않습니다)된 경우 이를 '
 '적용하지 아니할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 129},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000828',
              'chunk_char_len': 167,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
