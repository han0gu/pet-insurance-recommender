from langchain_core.documents import Document

chunk = Document(
    page_content=('6) 씹어먹는 기능과 말하는 기능 모두에 약간의 장해를 남긴 때 | 10\n'
 '7) 씹어먹는 기능 또는 말하는 기능에 약간의 장 해를 남긴 때 | 5\n'
 '8) 치아에 14개 이상의 결손이 생긴 때 | 20\n'
 '9) 치아에 7개 이상의 결손이 생긴 때 | 10\n'
 '10) 치아에 5개 이상의 결손이 생긴 때 | 5'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 181},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'head']},
 'indexing': {'chunk_id': 'chunk_000647',
              'chunk_char_len': 165,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
