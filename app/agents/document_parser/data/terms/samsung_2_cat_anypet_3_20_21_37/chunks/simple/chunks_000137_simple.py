from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 회사는 제1항 및 제2항을 위반하였을 경우에 새로이 증가 또는 교체되는 해당 보험의 목적에 대하 여는 보상하여 드리지 않습니다. '
 '④ 제1항에 따라 보험의 목적이 교체되는 경우에는 보험의 목적 교체전 계약과 동일한 보장조건 및 인수기준에 따라 가입될 수 있으며, '
 '보험의 목적 교체시점부터 잔여 보험기간(보험의 목적 교체전 계약의 보험기간 만료일)까지 보상하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000137',
              'chunk_char_len': 209,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
