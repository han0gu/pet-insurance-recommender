from langchain_core.documents import Document

chunk = Document(
    page_content=('10) 한 다리가 5cm 이상 짧아지거나 길어진 때 | 30\n'
 '11) 한 다리가 3cm 이상 짧아지거나 길어진 때 | 15\n'
 '12) 한 다리가 1cm 이상 짧아지거나 길어진 때 | 5\n'
 '나. | 장해판정기준'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 145},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000938',
              'chunk_char_len': 112,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
