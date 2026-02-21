from langchain_core.documents import Document

chunk = Document(
    page_content=('간에 대한 보험료의 자동대출납입을 위해서는 제1항에 따라\n'
 '재신청을 하여야 합니다.\n'
 '\uf000 보험료의 자동대출납입이 행하여진 경우에도 자동대출\n'
 '납입전 납입최고(독촉)기간이 끝나는 날의 다음날부터 1개\n'
 '월 이내에 계약자가 계약의 해지를 청구한 때에는 회사는\n'
 '보험료의 자동대출납입이 없었던 것으로 하여 제35조(해약\n'
 '환급금) 제1항에 따른 해약환급금을 지급합니다.\n'
 '\uf000 회사는 자동대출납입이 종료된 날부터 15일 이내에 자동\n'
 '대출납입이 종료되었음을 서면, 전화(음성녹음) 또는 전자'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000101',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
