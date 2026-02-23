from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 용 어 풀 이 중대한 과실 주의의무의 위반이 현저한 과실, 즉 현저한 부주의, 태만의 경우로서 조금만 주의를 하였다면 충분히 피해의 '
 '발생을 막을 수 있었음에도 그 주의조차 태만 | 용 어 풀 이 중대한 과실 주의의무의 위반이 현저한 과실, 즉 현저한 부주의, 태만의 '
 '경우로서 조금만 주의를 하였다면 충분히 피해의 발생을 막을 수 있었음에도 그 주의조차 태만 |\n'
 '# 히 한 높은 강도의 주의의무위반제9조(알릴 의무 위반의 효과)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000488',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
