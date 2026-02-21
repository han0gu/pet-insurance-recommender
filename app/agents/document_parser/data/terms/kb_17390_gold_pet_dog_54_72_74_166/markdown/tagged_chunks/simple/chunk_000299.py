from langchain_core.documents import Document

chunk = Document(
    page_content=('- 확대(다만, 유방암 환자의 유방재건술은 보장합니다)·축소술, 지방흡입\n'
 "- 술(다만, ｢국민건강보험법｣ 및 관련 고시에 따라 요양급여에 해당하는 '여\n"
 "- 성형 유방증'을 수술하면서 그 일련의 과정으로 시행한 지방흡입술은 보상\n"
 '- 합니다), 주름살제거술 등\n'
 '- 나. 사시교정, 안와격리증(양쪽 눈을 감싸고 있는 뼈와 뼈 사이의 거리가 넓\n'
 '- 은 증상)의 교정 등 시각계 수술로서 시력개선 목적이 아닌 외모개선 목\n'
 '- 적의 수술\n'
 '- 다. 안경, 콘택트렌즈 등을 대체하기 위한 시력교정술(국민건강보험 요양급여'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000299',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
