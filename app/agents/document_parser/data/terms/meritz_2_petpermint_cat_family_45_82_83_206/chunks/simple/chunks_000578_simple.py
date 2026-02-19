from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 이 특별약관에서 정한 보장개시일 이전에 발생한 질병에 대하여 계약을 무효로 하는 경우에도 제2조(특별면책조건의 내용) '
 '제1항에서 정한 특정질병에 대하여 면책을 조건으로 체결한 후 보장개시일 이전에 동일한 특정질병이 발생한 경 우에는 계약을 무효로 하지 '
 '않습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 166},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000578',
              'chunk_char_len': 150,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
