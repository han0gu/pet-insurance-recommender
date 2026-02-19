from langchain_core.documents import Document

chunk = Document(
    page_content='강동물에 실시하는 외과수술 및 기타 검사 또는 점 안, 귀청소 등의 관리 비용',
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 163},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye', 'other']},
 'indexing': {'chunk_id': 'chunk_000563',
              'chunk_char_len': 43,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
