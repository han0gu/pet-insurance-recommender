from langchain_core.documents import Document

chunk = Document(
    page_content=("회사는 제1항에서 정한 지급기일내에 보험금을 지급하지</p><footer id='34' "
 "style='font-size:14px'>89</footer><p id='35' data-category='paragraph' "
 "style='font-size:16px'>않았을 때(제2항에서 정한 지급예정일을 통지한 경우를 포<br>함합니다)에는 그 다음날부터 "
 '지급일까지의 기간에 대하여<br>【별표1(보험금을 지급할 때의 적립이율 계산)】에서 정한<br>이율로 계산한 금액을 보험금에 더하여 '
 '지급합니다'),
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
 'indexing': {'chunk_id': 'chunk_000303',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
