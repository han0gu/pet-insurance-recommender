from langchain_core.documents import Document

chunk = Document(
    page_content=('지급예정일을 통<br>지한 경우를 포함합니다)에는 그 다음날부터 지급일까지의<br>기간에 대하여 【별표1(보험금을 지급할 때의 적립이율 '
 '계<br>산)】에서 정한 이율로 계산한 금액을 보험금에 더하여 지<br>급합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000049',
              'chunk_char_len': 120,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
