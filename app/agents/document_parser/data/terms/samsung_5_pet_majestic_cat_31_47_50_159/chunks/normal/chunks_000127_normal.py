from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제1항 및 제2항에 의한 보험료의 자동대출납입 기간은 최초 자동대출납입일부터 1년 을 한도로 하며 그 이후의 기간에 대한 보험료의 '
 '자동대출납입을 위해서는 제1항에 따라 재신청을 하여야 합니다. ④ 보험료의 자동대출납입이 행하여진 경우에도 자동대출납입 전 '
 '납입최고(독촉)기간이 끝나는 날의 다음날부터 1개월 이내에 계약자가 계약의 해지를 청구한 때에는 회사는 보험료의 자동대출납입이 없었던 '
 '것으로 하여 제36조(해약환급금) 제1항에 따른 해약 환급금을 지급합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 42},
 'term_type': 'basic',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000127',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
