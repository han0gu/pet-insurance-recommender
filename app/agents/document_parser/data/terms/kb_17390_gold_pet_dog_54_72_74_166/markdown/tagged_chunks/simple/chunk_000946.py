from langchain_core.documents import Document

chunk = Document(
    page_content=('| 유형 이동동작 | 3) 목발 또는 보행기(walker)를 사용하지 않으면 독립적 20% 인 보행이 불가능한 상태 | 특별 약 |\n'
 '| 유형 이동동작 | 4) 보조기구 없이 독립적인 보행은 가능하나 보행시 파행 (절뚝거림)이 있으며, 난간을 잡지 않고는 계단을 오 '
 '10% 르내리기가 불가능한 상태 또는 평지에서 100m 이상을 걷지 못하는 상태 | 관 |\n'
 '| 음식물 섭취 | 1) 입으로 식사를 전혀 할 수 없어 계속적으로 튜브(비위 관 또는 위루관)나 경정맥 수액을 통해 부분 혹은 전 '
 '20% 적인 영양공급을 받는 상태 | 별 표 |'),
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
 'indexing': {'chunk_id': 'chunk_000946',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
