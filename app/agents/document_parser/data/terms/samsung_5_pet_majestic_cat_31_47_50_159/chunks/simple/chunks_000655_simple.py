from langchain_core.documents import Document

chunk = Document(
    page_content=('4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관 제27조 (특별약관의 재 가입에 관한 사항) 제5항에 따라 보험계약이 '
 '연장된 경우에는 종전 계약의 보험기간 을 연장하는 것으로 보아 제2항을 적용하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 108},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'other']},
 'indexing': {'chunk_id': 'chunk_000655',
              'chunk_char_len': 124,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
