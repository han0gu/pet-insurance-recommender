from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자, 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실\n'
 '【중과실(중대한 과실)】\n'
 '주의의무의 위반이 현저한 과실,「중대한 과실」, 즉 현저한 부주의, 태만의 경우 로서 조금만 주의를 하였다면 충분히 피해의 발생을 막을 '
 '수 있었음에도 그 주의 조차 태만히 한 높은 강도의 주의의무 위반(이하 같습니다.)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 4},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000020',
              'chunk_char_len': 178,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
