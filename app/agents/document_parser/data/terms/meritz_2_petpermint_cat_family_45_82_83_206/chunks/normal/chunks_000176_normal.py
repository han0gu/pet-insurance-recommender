from langchain_core.documents import Document

chunk = Document(
    page_content=('【 현저하게 공정을 잃은 합의 】\n'
 '사회통념상 일반 보통인이라면 그 같은 일을 하지 않을 정도로 현저하게 공정성을 잃은 것을 말합니다.\n'
 '제46조(개인정보보호)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 81},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000176',
              'chunk_char_len': 87,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
