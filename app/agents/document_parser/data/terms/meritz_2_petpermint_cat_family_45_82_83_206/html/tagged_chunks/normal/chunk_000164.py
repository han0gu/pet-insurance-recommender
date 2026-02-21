from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 계약자가 제1항 제5호에 따라 보험가입금액을 감<br>액하고자 할 때에는 그 감액된 부분은 해지된 것으로 보며,<br>이로써 '
 '회사가 지급하여야 할 해약환급금이 있을 때에는 제<br>35조(해약환급금) 제1항에 따른 해약환급금을 '
 "계약자에게<br>지급합니다.</p><br><h1 id='20' style='font-size:20px'>【 감액 】</h1><br><p "
 "id='21' data-category='paragraph' style='font-size:16px'>보험료, 보험금, 계약자적립액 등을 "
 '산정하는 기준이 되<br>는'),
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
 'indexing': {'chunk_id': 'chunk_000164',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
