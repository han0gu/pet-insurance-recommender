from langchain_core.documents import Document

chunk = Document(
    page_content=('. 1. 1 시행)를 말하며 이후 한국표준질병·사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관에서 보장하는 질병(상병) 해당 '
 '여부를 판단합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 50},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 84,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
