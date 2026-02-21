from langchain_core.documents import Document

chunk = Document(
    page_content=("- 3. 제1조(보상하는 손해) 제1항 제2호 '다'목 또는 '라'목의 비용: 이 비용과 제1호에 의한 보상액\n"
 '- 의 합계액을 보상한도액 내에서 보상합니다.\n'
 '② 보험기간 중 발생하는 사고에 대한 회사의 보상총액은 보험증권에 기재된 총 보상한도액을 한도로\n'
 '합니다# 제5조(의무보험과의 관계)- ① 회사는 이 약관에 의하여 보상하여야 하는 금액이 의무보험에서 보상하는 금액을 초과할 때에 '
 '한하여\n'
 '- 그 초과액만을 보상합니다. 다만, 의무보험이 다수인 경우에는 제6조(보험금의 분담)를 따릅니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000114',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
