from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 회사는 제1항 및 제2항을 위반하였을 경우에 새로이 증가 또는 교체되는 해당피보험자에 대하여는 보상하여 드리지 않습니다. ④ '
 '제1항에 따라 피보험자가 교체되는 경우에는 피보험자 교체 전 계약과 동일한 보장조건 및 인수기 준에 따라 가입될 수 있으며, 피보험자 '
 '교체시점부터 잔여 보험기간(피보험자 교체 전 계약의 보험 기간 만료일)까지 보상하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 40},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000199',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
