from langchain_core.documents import Document

chunk = Document(
    page_content=('- 있습니다.\n'
 '- ② 회사는 계약자 또는 피보험자의 고의로 손해가 발생한 경우 이 계약을 해지할 수 있습니다.\n'
 '- ③ 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 관계없이 그 사실을 안 날부터 1개\n'
 '- 월 이내에 이 계약을 해지할 수 있습니다.\n'
 '1. 계약자, 피보험자 또는 이들의 대리인이 제12조(계약 전 알릴 의무)에도 불구하고 고의 또는 중\n'
 '대한 과실로 중요한 사항에 대하여 사실과 다르게 알린 때.【고의】 자기의 행위가 불법구성요건을 실현함을 인식하고 인용하는 행위자의 심적 '
 '태도를 말합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000070',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
