from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 제1항 및 제2항에 따른 보험료의 자동대출납입 기간은 최초 자동대출납입일부터 1년을 한도로 하며 그 이후의 기 간에 대한 '
 '보험료의 자동대출납입을 위해서는 제1항에 따라 재신청을 하여야 합니다. \uf000 보험료의 자동대출납입이 행하여진 경우에도 자동대출 '
 '납입전 납입최고(독촉)기간이 끝나는 날의 다음날부터 1개 월 이내에 계약자가 계약의 해지를 청구한 때에는 회사는 보험료의 자동대출납입이 '
 '없었던 것으로 하여 제35조(해약 환급금) 제1항에 따른 해약환급금을 지급합니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 76},
 'term_type': 'basic',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000126',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
