from langchain_core.documents import Document

chunk = Document(
    page_content=("- 여성형 유방증'을 수술하면서 그 일련의 과정으로 시행한 지방흡입술은\n"
 '- 보상합니다), 주름살제거술 등\n'
 '- 나. 사시교정, 안와격리증(양쪽 눈을 감싸고 있는 뼈와 뼈 사이의 거리가 넓'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000266',
              'chunk_char_len': 103,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
