from langchain_core.documents import Document

chunk = Document(
    page_content=('부 가 설 명손가락![image](/image/placeholder)\n'
 '150 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- 150 -# 11. 발가락의장해| 가. 장해의 분류 |  |\n'
 '| --- | --- |\n'
 '| 장해의 분류 | 지급률 |\n'
 '| 1) 한 발의 리스프랑관절 이상을 잃었을 때 | 40 |\n'
 '| 2) 한 발의 5개 발가락을 모두 잃었을 때 | 30 |\n'
 '| 3) 한 발의 첫째 발가락을 잃었을 때 4) 한 발의 첫째 발가락 이외의 발가락을 잃었을 때 (발가락 하나마다) | 10 5 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000917',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
