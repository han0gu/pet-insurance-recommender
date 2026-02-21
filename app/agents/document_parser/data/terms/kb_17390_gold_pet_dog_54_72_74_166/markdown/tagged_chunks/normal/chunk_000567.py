from langchain_core.documents import Document

chunk = Document(
    page_content=('수의사법 제17조(개설)에서 규정한 국내의 동물병원에서 수의사의 관리 하에 직접적\n'
 '인 치료를 목적으로 기구를 사용하여 생체에 절단, 절제 등의 조작을 가하는 것을 말 상- 합니다. 단, 수술에서 아래에 정한 사항은 '
 '제외합니다.\n'
 '- 1. 흡인(吸引)\n'
 '- 2. 천자(穿刺) 등의 조치\n'
 '- 3. 미용성형 목적의 수술\n'
 '- 4. 검사 및 진단을 위한 수술(생검(生檢), 복강경검사(腹腔鏡檢査) 등)\n'
 '별약해| 용 어 풀 이 | 질 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000567',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
