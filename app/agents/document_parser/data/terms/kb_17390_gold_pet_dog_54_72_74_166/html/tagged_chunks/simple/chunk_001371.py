from langchain_core.documents import Document

chunk = Document(
    page_content=('. 갱신된 특별약관(이하 "갱신보장특약"이라 합니다)의 만기일이 회사가 정한</h1><br><p id=\'19\' '
 "data-category='list' style='font-size:14px'>기간 내일 것<br>2. 갱신일에 있어서 피보험자의 연령 "
 '또는 피보험자의 반려동물 연령이 회사가 정<br>한 연령의 범위 내일 것<br>3'),
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
 'indexing': {'chunk_id': 'chunk_001371',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
