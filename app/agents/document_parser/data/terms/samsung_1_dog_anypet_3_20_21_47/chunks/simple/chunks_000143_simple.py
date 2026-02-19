from langchain_core.documents import Document

chunk = Document(
    page_content=('② 보험기간 중 발생하는 사고에 대한 회사의 보상총액은 보험증권에 기재된 총 보상한도액을 한도로 합니다\n'
 '제5조(의무보험과의 관계)\n'
 '① 회사는 이 약관에 의하여 보상하여야 하는 금액이 의무보험에서 보상하는 금액을 초과할 때에 한하여 그 초과액만을 보상합니다. 다만, '
 '의무보험이 다수인 경우에는 제6조(보험금의 분담)를 따릅니다. ② 제1항의 의무보험은 피보험자가 법률에 의하여 의무적으로 가입하여야 하는 '
 '보험으로서 공제계약 을 포함합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 26},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000143',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
