from langchain_core.documents import Document

chunk = Document(
    page_content=('. 장해의 분류</td><td></td></tr></thead><tbody><tr><td>장해의 '
 '분류</td><td>지급률</td></tr><tr><td>1) 씹어먹는 기능과 말하는 기능 모두에 심한 장해를 남긴 '
 '때</td><td>100</td></tr><tr><td>2) 씹어먹는 기능에 심한 장해를 남긴 '
 '때</td><td>80</td></tr><tr><td>3) 말하는 기능에 심한 장해를 남긴 '
 '때</td><td>60</td></tr><tr><td>4) 씹어먹는 기능과 말하는 기능 모두에 뚜렷한 장해를 남긴'),
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
 'indexing': {'chunk_id': 'chunk_001514',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
