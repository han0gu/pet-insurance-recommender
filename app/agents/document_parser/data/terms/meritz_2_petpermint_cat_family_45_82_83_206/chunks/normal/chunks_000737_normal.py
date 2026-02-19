from langchain_core.documents import Document

chunk = Document(
    page_content=('또는 상해를 입은 후 18개월이 지난 후에 판정 함을 원칙으로 한다. 단, 질병발생 또는 상해를 입은 후 의식상실이 1개월 이상 지속된 '
 '경우에 는 질병발생 또는 상해를 입은 후 12개월이 지난 후에 판정할 수 있다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 202},
 'term_type': 'special',
 'clause': {'clause_type': 'waiting', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000737',
              'chunk_char_len': 120,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
