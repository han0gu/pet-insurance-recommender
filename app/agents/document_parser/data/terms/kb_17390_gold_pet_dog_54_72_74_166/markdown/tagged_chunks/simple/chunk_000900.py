from langchain_core.documents import Document

chunk = Document(
    page_content=('# 급률은 60% 한도로 한다.- \n'
 '# 9. 다리의 장해| 가. |  |\n'
 '| --- | --- |\n'
 '| 장해의 분류 장해의 분류 | 지급률 |\n'
 '| 1) 두 다리의 발목 이상을 잃었을 때 | 100 |\n'
 '| 2) 한 다리의 발목 이상을 잃었을 때 3) 한 다리의 3대 관절 중 관절 하나의 기능을 완전히 잃었을 때 30 | 60 |\n'
 '| 4) 한 다리의 3대 관절 중 관절 하나의 기능에 심한 장해를 남긴 때 20 |  |\n'
 '| 5) 한 다리의 3대 관절 중 관절 하나의 기능에 뚜렷한 장해를 남긴 때 10 |  |'),
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
 'indexing': {'chunk_id': 'chunk_000900',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
