from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 지급률의 결정\n'
 '1) 한 팔의 3대 관절중 관절 하나에 기능장해가 생기고 다른 관절 하나에 기능장해가 발생한 경우 지급률은 각각 적용하여 합산한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 192},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000696',
              'chunk_char_len': 84,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
