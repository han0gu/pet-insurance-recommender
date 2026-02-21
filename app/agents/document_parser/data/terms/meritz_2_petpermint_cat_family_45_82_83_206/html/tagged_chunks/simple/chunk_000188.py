from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자<br>에게 지급할 기타 모든 지급금의 합계액에서 계약자의 회사<br>에 대한 모든 채무액을 뺀 금액을 초과하는 경우에는 '
 '보험<br>료의 자동대출납입을 더는 할 수 없습니다.<br>\uf000 제1항 및 제2항에 따른 보험료의 자동대출납입 기간은<br>최초 '
 '자동대출납입일부터 1년을 한도로 하며 그 이후의 기<br>간에 대한 보험료의 자동대출납입을 위해서는 제1항에 따라<br>재신청을 하여야 '
 '합니다.<br>\uf000 보험료의 자동대출납입이 행하여진 경우에도 자동대출<br>납입전 납입최고(독촉)기간이 끝나는 날의 다음날부터 '
 '1개<br>월 이내에'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000188',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
