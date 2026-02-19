from langchain_core.documents import Document

chunk = Document(
    page_content='4) "씹어먹는 기능에 약간의 장해를 남긴 때" 라 함은 아래의 경우 중 하나 이상 에 해당되는 때를 말한다.',
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'other',
            'risk_domains': ['dental', 'head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000894',
              'chunk_char_len': 61,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
