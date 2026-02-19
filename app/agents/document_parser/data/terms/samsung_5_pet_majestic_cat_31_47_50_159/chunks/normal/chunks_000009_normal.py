from langchain_core.documents import Document

chunk = Document(
    page_content=('⑥ [갱신형] 특별약관의 갱신 관련 용어\n'
 '1. 최초계약: [갱신형] 특별약관이 최초로 부가되는 경우를 말합니다. 2. 갱신계약: [갱신형] 특별약관의 보험기간이 끝난 후 제도성 '
 '특별약관 「5-1. [갱신'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 31},
 'term_type': 'basic',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000009',
              'chunk_char_len': 113,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
