from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 제22조(계약의 무효) 제1항 제2호의 경우에<br>는 실제 만 나이를 적용합니다.<br>\uf000 제1항의 보험나이는 '
 '계약일 현재 피보험자의 실제 만 나<br>이를 기준으로 6개월 미만의 끝수는 버리고 6개월 이상의<br>끝수는 1년으로 하여 계산하며, '
 '이후 매년 계약해당일에 나<br>이가 증가하는 것으로 합니다.<br>\uf000 피보험자의 나이 또는 성별에 관한 청약서상 '
 "기재사항이</p><footer id='25' style='font-size:14px'>69</footer><p id='26'"),
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
 'indexing': {'chunk_id': 'chunk_000168',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
