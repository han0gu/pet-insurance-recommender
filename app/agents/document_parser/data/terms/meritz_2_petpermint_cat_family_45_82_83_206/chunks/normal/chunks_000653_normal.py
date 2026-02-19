from langchain_core.documents import Document

chunk = Document(
    page_content=('구(입을 벌림)상태에서 위‧아래턱(상ㆍ하악)의 가운 데 앞니(중절치)간 거리를 기준으로 한다. 단, 가 운데 앞니(중절치)가 없는 '
 '경우에는 측정가능한 인 접 치아간 거리의 최대치를 기준으로 한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 183},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000653',
              'chunk_char_len': 109,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
