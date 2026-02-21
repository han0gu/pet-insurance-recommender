from langchain_core.documents import Document

chunk = Document(
    page_content=('씹어먹는 기능과 말하는 기능 모두에 뚜렷한 장해를 남긴 때</td><td>40</td></tr><tr><td>5) 씹어먹는 기능 또는 '
 '말하는 기능에 뚜렷한 장해를 남긴 때</td><td>20</td></tr><tr><td>6) 씹어먹는 기능과 말하는 기능 모두에 약간의 '
 '장해를 남긴 때</td><td>10</td></tr><tr><td>7) 씹어먹는 기능 또는 말하는 기능에 약간의 장해를 남긴 '
 '때</td><td>5</td></tr><tr><td>8) 치아에 14개 이상의 결손이 생긴'),
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
 'indexing': {'chunk_id': 'chunk_001515',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
