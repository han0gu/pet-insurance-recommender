from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 지급예정일은 다음 각 호의 어느 하나에 해당하는 경우를 제외하고는 제8조(보험금의 청구)에서 정한 서류를 접수한 날부터 '
 '30영업일 이내에서 정합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 100},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000569',
              'chunk_char_len': 89,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
