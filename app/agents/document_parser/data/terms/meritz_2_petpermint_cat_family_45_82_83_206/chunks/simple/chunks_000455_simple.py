from langchain_core.documents import Document

chunk = Document(
    page_content=('및 해부검사, 장례비, 이장비 등 사후에 필요한 비용 ⑰ 마이크로 칩 이식 비용, 각종 증빙서류의 작성비용 (우송비 포함) ⑱ '
 '과잉진료행위로 인한 비용'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000455',
              'chunk_char_len': 84,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
