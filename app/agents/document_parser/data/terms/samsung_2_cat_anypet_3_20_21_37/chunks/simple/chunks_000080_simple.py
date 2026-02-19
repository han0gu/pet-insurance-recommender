from langchain_core.documents import Document

chunk = Document(
    page_content=('제6관 계약의 해지 및 보험료의 환급 등\n'
 '제26조(계약의 해지)\n'
 '① 계약자는 손해가 발생하기 전에는 언제든지 계약을 해지할 수 있습니다. 다만, 타인을 위한 계약의 경우에는 계약자는 그 타인의 동의를 '
 '얻거나 보험증권을 소지한 경우에 한하여 계약을 해지할 수 있습니다. ② 회사는 계약자 또는 피보험자의 고의로 손해가 발생한 경우 이 '
 '계약을 해지할 수 있습니다. ③ 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 관계없이 그 사실을 안 날부터 1개 월 이내에 '
 '이 계약을 해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000080',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
