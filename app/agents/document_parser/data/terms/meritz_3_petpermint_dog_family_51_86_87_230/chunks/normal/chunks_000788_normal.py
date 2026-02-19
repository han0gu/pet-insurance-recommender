from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그중 심장에서 가까운 쪽부터 중수지관절, 지관절이 라 한다. 4) 다른 네 손가락에는 3개의 손가락관절이 있다. 그 중 심장에서 '
 '가까운 쪽부터 중수지관절, 제1지관절 (근위지관절) 및 제2지관절(원위지관절)이라 부른다. 5) “손가락을 잃었을 때”라 함은 첫째 '
 '손가락에서는 지 관절부터 심장에서 가까운 쪽에서, 다른 네 손가락에서 는 제1지관절(근위지관절)부터(제1지관절 포함) 심장 에서 가까운 '
 '쪽으로 손가락이 절단되었을 때를 말한 다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 221},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000788',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
