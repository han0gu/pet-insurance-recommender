from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 아래에 열거된 국민건강보험 비급여 대상으로 신체의 필수 기능개선 목적이 아닌\n'
 '# 외모개선 목적의 치료를 위한 수술- 가. 쌍꺼풀수술(이중검수술. 다만, 안검하수, 안검내반 등을 치료하기 위한 시력개\n'
 '- 선 목적의 이중검수술은 보상합니다), 코성형수술(융비 술), 유방확대(다만, 유\n'
 '- 방암 환자의 유방재건술은 보상합니다)·축소술, 지방흡입술(다만, 「국민건강보\n'
 "- 험법」 및 관련 고시에 따라 요양급여에 해당하는 '여성형 유방증'을 수술하면\n"
 '- 서 그 일련의 과정으로 시행한 지방흡입술은 보상합니다), 주름살 제거술 등'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000332',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
