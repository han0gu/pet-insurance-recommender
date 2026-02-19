from langchain_core.documents import Document

chunk = Document(
    page_content=('. [보험료 할증] 일반적인 경우보다 위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 보험 가입 후 기간이 경 과함에 따라 위험의 '
 '크기 및 정도가 점차 증가하는 위험 또는 기간의 경과에 상관없이 일정한 상태 를 유지하는 위험에 적용하는 방법으로 위험 정도에 따라 '
 '특별보험료를 추가로 부가하는 방법을 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 56},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000243',
              'chunk_char_len': 175,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
