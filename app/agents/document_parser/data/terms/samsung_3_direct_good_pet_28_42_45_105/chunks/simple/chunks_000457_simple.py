from langchain_core.documents import Document

chunk = Document(
    page_content=('지급기준 | 1회당 보상한도액\n'
 '이물제거를 목적으로 내시경을 받은 경우 | 200만원\n'
 '이물제거를 목적으로 구토유도약물을 투약한 경우 | 20만원'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000457',
              'chunk_char_len': 79,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
