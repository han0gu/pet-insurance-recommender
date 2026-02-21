from langchain_core.documents import Document

chunk = Document(
    page_content=('. 쌍꺼풀수술(이중검수술. 다만, 안검하수, 안검내반 등을 치료하기 위한<br>시력개선 목적의 이중검수술은 보장합니다), '
 '코성형수술(융비술), 유방<br>확대(다만, 유방암 환자의 유방재건술은 보장합니다)·축소술, 지방흡입<br>술(다만, ｢국민건강보험법｣ '
 "및 관련 고시에 따라 요양급여에 해당하는 '<br>여성형 유방증'을 수술하면서 그 일련의 과정으로 시행한 지방흡입술은<br>보상합니다), "
 '주름살제거술 등<br>나'),
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
 'indexing': {'chunk_id': 'chunk_000405',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
