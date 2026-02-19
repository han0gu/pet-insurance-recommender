from langchain_core.documents import Document

chunk = Document(
    page_content=('가. 쌍꺼풀수술(이중검수술. 다만, 안검하수, 안검내반 등을 치료하기 위한 시력개 선 목적의 이중검수술은 보상합니다), 코성형수술(융비 '
 '술), 유방확대(다만, 유 방암 환자의 유방재건술은 보상합니다)·축소술, 지방흡입술(다만, 「국민건강보 험법」 및 관련 고시에 따라 '
 "요양급여에 해당하는 '여성형 유방증'을 수술하면 서 그 일련의 과정으로 시행한 지방흡입술은 보상합니다), 주름살 제거술 등 나"),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['eye', 'skin', 'other']},
 'indexing': {'chunk_id': 'chunk_000394',
              'chunk_char_len': 220,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
