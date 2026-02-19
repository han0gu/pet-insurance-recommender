from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 제1항에도 불구하고 계약일부터 10년 이내에 인출하는 경우, 각 인출시점까지의 인출 금액 총합계는 이미 납입한 보험료를 초과할 수 '
 '없습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 34},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000041',
              'chunk_char_len': 82,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
