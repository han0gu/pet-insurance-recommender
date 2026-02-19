from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자, 피보험자 또는 이들의 대리인이 제12조(계약 전 알릴 의무)에도 불구하고 고의 또는 중 대한 과실로 중요한 사항에 대하여 '
 '사실과 다르게 알린 때.\n'
 '【고의】 자기의 행위가 불법구성요건을 실현함을 인식하고 인용하는 행위자의 심적 태도를 말합니다. 【중대한 과실(중과실)】 주의의무의 '
 '위반이 현저한 과실, 즉 현저한 부주의, 태만의 경우로서 조금만 주의를 하였다면 충분히 피해의 발생을 막을 수 있었음에도 그 주의조차 '
 '태만히 한 높은 강도의 주의의 무위반을 말합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000085',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
