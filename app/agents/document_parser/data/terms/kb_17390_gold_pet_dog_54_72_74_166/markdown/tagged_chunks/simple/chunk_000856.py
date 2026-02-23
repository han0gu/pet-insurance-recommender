from langchain_core.documents import Document

chunk = Document(
    page_content=('- \n'
 '- 142 -4. 씹어먹거나# 말하는 장해| 가. 장해의 분류 |  |\n'
 '| --- | --- |\n'
 '| 장해의 분류 | 지급률 |\n'
 '| 1) 씹어먹는 기능과 말하는 기능 모두에 심한 장해를 남긴 때 | 100 |\n'
 '| 2) 씹어먹는 기능에 심한 장해를 남긴 때 | 80 |\n'
 '| 3) 말하는 기능에 심한 장해를 남긴 때 | 60 |\n'
 '| 4) 씹어먹는 기능과 말하는 기능 모두에 뚜렷한 장해를 남긴 때 | 40 |\n'
 '| 5) 씹어먹는 기능 또는 말하는 기능에 뚜렷한 장해를 남긴 때 | 20 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000856',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
