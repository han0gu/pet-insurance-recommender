from langchain_core.documents import Document

chunk = Document(
    page_content=('자동대출<br>납입전 납입최고(독촉)기간이 끝나는 날의 다음날부터 1개<br>월 이내에 계약자가 계약의 해지를 청구한 때에는 '
 '회사는<br>보험료의 자동대출납입이 없었던 것으로 하여 제35조(해약<br>환급금) 제1항에 따른 해약환급금을 '
 '지급합니다.<br>\uf000 회사는 자동대출납입이 종료된 날부터 15일 이내에 자동<br>대출납입이 종료되었음을 서면, 전화(음성녹음) '
 "또는 전자<br>문서(SMS 포함) 등으로 계약자에게 안내하여 드립니다.</p><br><h1 id='53'"),
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
 'indexing': {'chunk_id': 'chunk_000189',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
