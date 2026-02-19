from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이 상 지난 유효한 계약으로서 그 보험종목의 변경을 요청할 때에는 회사의 '
 '사업방법서에서 정하는 방법에 따라 이를 변 경하여 드립니다. \uf000 회사는 계약자가 제1항 제5호에 따라 보험가입금액을 감 액하고자 '
 '할 때에는 그 감액된 부분은 해지된 것으로 보며, 이로써 회사가 지급하여야 할 해약환급금이 있을 때에는 제 35조(해약환급금) 제1항에 '
 '따른 해약환급금을 계약자에게 지급합니다.\n'
 '【 감액 】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 69},
 'term_type': 'basic',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000108',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
