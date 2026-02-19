from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험설계사 등의 행위가 없 었다 하더라도 계약자 또는 피보험자가 사실대로 알리 지 않거나 부실한 사항을 알렸다고 인정되는 '
 '경우에는 계약을 해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 61},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000068',
              'chunk_char_len': 94,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
