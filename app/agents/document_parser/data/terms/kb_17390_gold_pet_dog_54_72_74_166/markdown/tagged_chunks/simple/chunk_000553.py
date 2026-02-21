from langchain_core.documents import Document

chunk = Document(
    page_content=('|  |  |\n'
 '| ![image](/image/placeholder)\n'
 '계약일 보장개시일\n'
 '30일\n'
 '2024년 4월 10일 2024년 5월 9일 - 단, 상해를 직접적인 원인으로 치료를 받은 경우에는 보장개시일은 보험계약 일로 합니다. '
 '예 시 2 슬관절/고관절 탈구의 보장개시일 |  |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000553',
              'chunk_char_len': 157,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
