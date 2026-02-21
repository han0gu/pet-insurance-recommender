from langchain_core.documents import Document

chunk = Document(
    page_content=('주의를 하였다면 충분히 피해의 발생을 막을 수 있었음에도 그 주의조차 태만히 한 높은 강도의 주의의\n'
 '무위반을 말합니다.2. 뚜렷한 위험의 변경 또는 증가와 관련된 제13조(계약 후 알릴 의무)에서 정한 계약 후 알릴 의\n'
 '무를 계약자, 피보험자 또는 이들의 대리인이 이행하지 않았을 때④ 제3항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 '
 '회사는 계약을 해지할 수 없\n'
 '습니다.- 1. 회사가 계약 당시에 그 사실을 알았거나 과실로 인하여 알지 못하였을 때'),
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
 'indexing': {'chunk_id': 'chunk_000069',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
