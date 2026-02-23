from langchain_core.documents import Document

chunk = Document(
    page_content=('- (1년미만의 기간에 적용되는 요율)로 계산한 보험료를 뺀 잔액. 다만, 계약자, 피보험자의 고의\n'
 '- 또는 중대한 과실로 무효가 된 때에는 보험료를 돌려드리지 않습니다.\n'
 '- ② 보험기간이 1년을 초과하는 계약이 무효 또는 효력상실인 경우에는 무효 또는 효력상실의 원인이\n'
 '- 생긴 날 또는 해지일이 속하는 보험년도의 보험료는 위 제1항의 규정을 적용하고 그 이후의 보험\n'
 '- 년도에 속하는 보험료는 전액을 돌려드립니다.\n'
 "- 제1항 제2호에서 '계약자 또는 피보험자의 책임 있는 사유'라 함은 다음 각호를 말합니다."),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000081',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
