from langchain_core.documents import Document

chunk = Document(
    page_content='제5조 (수술의 정의와 장소)',
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 73},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000386',
              'chunk_char_len': 16,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
