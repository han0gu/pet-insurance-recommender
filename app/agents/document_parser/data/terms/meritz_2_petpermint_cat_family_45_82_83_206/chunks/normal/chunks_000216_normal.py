from langchain_core.documents import Document

chunk = Document(
    page_content='. 다만, 보험료 갱신형 계약 등 일부 보험계약의 경 우 분납이 제한될 수 있습니다.',
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 92},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000216',
              'chunk_char_len': 47,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
