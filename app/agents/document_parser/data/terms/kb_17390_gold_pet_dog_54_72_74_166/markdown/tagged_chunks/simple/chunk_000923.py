from langchain_core.documents import Document

chunk = Document(
    page_content=('별 표 법 ㆍ 규정 |\n'
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 151- 151 -부 가 설 '
 '명발가락![image](/image/placeholder)\n'
 '![image](/image/placeholder)\n'
 '152 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)12.# 흉․복부장기 및# 비뇨생식기의 장해| 가. 장해의 분류 |  |\n'
 '| --- | --- |\n'
 '| 장해의 분류 | 지급률 |\n'
 '| 1) 심장 기능을 잃었을 때 | 100 |\n'
 '| 2) 흉복부장기 또는 비뇨생식기 기능을 잃었을 때 | 75 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000923',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
