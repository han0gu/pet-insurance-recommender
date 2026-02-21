from langchain_core.documents import Document

chunk = Document(
    page_content=('# 음식물의 상태(부패, 감염 여부 등)와 상관없이 모두 포함됩니다.| 3. 지급금과 용 | 이자율 관련 용어 어 정 의 |\n'
 '| --- | --- |\n'
 '| 연단위 | 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를 원금에 더한 금액을 다음 1년의 원금으로 하는 이자 '
 '계산방법을 말합니다. 복리 원금 100원, 이자율 연 10%를 가정할 때 - 1년 후 : 100원 + (100원 × 10%) = 110원 '
 '- 2년 후 : 110원 + (110원 × 10%) = 121원 |'),
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
 'indexing': {'chunk_id': 'chunk_000450',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
