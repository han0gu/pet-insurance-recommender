from langchain_core.documents import Document

chunk = Document(
    page_content='2) 정신행동\n가) 정신행동장해는 보험기간중에 발생한 뇌의 질병',
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 201},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000736',
              'chunk_char_len': 35,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
