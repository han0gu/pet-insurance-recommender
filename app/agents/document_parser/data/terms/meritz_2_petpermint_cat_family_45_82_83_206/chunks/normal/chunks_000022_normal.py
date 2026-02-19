from langchain_core.documents import Document

chunk = Document(
    page_content='(보험금의 지급사유)의 상해 관련 보험금 지급사유가 발생 한 때에는 해당 보험금을 지급하지 않습니다.',
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 52},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000022',
              'chunk_char_len': 56,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
