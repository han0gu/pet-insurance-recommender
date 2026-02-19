from langchain_core.documents import Document

chunk = Document(
    page_content=('1년이상 2년미만 | 60% | 50% | 40% | 30%\n'
 '2년이상 3년미만 | 75% | 60% | 45%\n'
 '3년이상 4년미만 | 80% | 60%\n'
 '4년이상 5년미만 | 80%'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 127},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000813',
              'chunk_char_len': 99,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
