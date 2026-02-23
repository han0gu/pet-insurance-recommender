from langchain_core.documents import Document

chunk = Document(
    page_content=('- 생긴 날 또는 해지일이 속하는 보험년도의 보험료는 위 제1항의 규정을 적용하고 그 이후의 보험\n'
 '- 년도에 속하는 보험료는 전액을 돌려드립니다.\n'
 "- 3 제1항 제2호에서 '계약자 또는 피보험자의 책임 있는 사유'라 함은 다음 각호를 말합니다.\n"
 '- 1. 계약자 또는 피보험자가 임의 해지하는 경우\n'
 '- 2. 회사가 제14조(사기에 의한 계약), 제26조(계약의 해지) 또는 제27조(중대사유로 인한 해지)에\n'
 '- 따라 계약을 취소 또는 해지하는 경우\n'
 '- 3. 보험료 미납으로 인한 계약의 효력 상실'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000080',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
