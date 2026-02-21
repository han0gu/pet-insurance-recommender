from langchain_core.documents import Document

chunk = Document(
    page_content=('- 검사, 불임수술, 불임복원술, 보조생식술(체내, 체외 인공수정을 포함합니\n'
 '- 다), 성장촉진과 관련된 수술\n'
 '- 3. 아래에 열거된 국민건강보험 비급여 대상으로 신체의 필수 기능개선 목적이\n'
 '- 아닌 외모개선 목적의 치료를 위한 수술\n'
 '- 가. 쌍꺼풀수술(이중검수술. 다만, 안검하수, 안검내반 등을 치료하기 위한\n'
 '- 시력개선 목적의 이중검수술은 보장합니다), 코성형수술(융비술), 유방\n'
 '- 확대(다만, 유방암 환자의 유방재건술은 보장합니다)·축소술, 지방흡입'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000298',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
