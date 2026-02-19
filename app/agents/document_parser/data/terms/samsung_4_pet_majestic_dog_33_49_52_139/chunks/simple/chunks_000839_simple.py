from langchain_core.documents import Document

chunk = Document(
    page_content=('1년이상 2년미만 | 60% | 50% | 40% | 30%\n'
 '2년이상 3년미만 | 75% | 60% | 45%\n'
 '3년이상 4년미만 | 80% | 60%\n'
 '4년이상 5년미만 | 80%'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 134},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000839',
              'chunk_char_len': 99,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
