from langchain_core.documents import Document

chunk = Document(
    page_content=('| 5) 씹어먹는 기능 또는 말하는 기능에 뚜렷한 장해를 남긴 때 | 20 |\n'
 '| 6) 씹어먹는 기능과 말하는 기능 모두에 약간의 장해를 남긴 때 | 10 |\n'
 '| 7) 씹어먹는 기능 또는 말하는 기능에 약간의 장해를 남긴 때 | 5 |\n'
 '| 8) 치아에 14개 이상의 결손이 생긴 때 | 20 |\n'
 '| 9) 치아에 7개 이상의 결손이 생긴 때 | 10 |\n'
 '| 10) 치아에 5개 이상의 결손이 생긴 때 | 5 |\n'
 '# 나. 장해의# 평가기준1) 씹어먹는 기능의 장해는 윗니(상악치아)와 아랫니(하악치아)의 맞물림'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000857',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
