from langchain_core.documents import Document

chunk = Document(
    page_content=('3) “발가락을 잃었을 때”라 함은 첫째 발가락에서는 지관절부터 심장에 가까운 쪽을, 나머지 네 발가락에 서는 '
 '제1지관절(근위지관절)부터(제1지관절 포함) 심 장에서 가까운 쪽을 잃었을 때를 말한다. 4) 리스프랑 관절 이상에서 잃은 때라 함은 '
 '족근-중족골 간 관절 이상에서 절단된 경우를 말한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 198},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000719',
              'chunk_char_len': 165,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
