from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 계약을 체결할 때 계약에서 정한 피보험자의 나이에 미달되었거나 초과되었을 경우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 '
 '계약나이에 도달한 경우에 는 유효한 계약으로 보나, 제2호의 만15세 미만자에 관한 예외가 인정되는 것은 아닙니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 67},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000100',
              'chunk_char_len': 140,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
