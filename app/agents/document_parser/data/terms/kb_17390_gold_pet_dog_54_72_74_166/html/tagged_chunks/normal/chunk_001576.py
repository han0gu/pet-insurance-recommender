from langchain_core.documents import Document

chunk = Document(
    page_content=('. 장해의 분류</td><td>공 통</td></tr></thead><tbody><tr><td>장해의 분류</td><td>지급률 '
 '사항</td></tr><tr><td>1) 두 팔의 손목 이상을 잃었을 때</td><td>100</td></tr><tr><td>2) 한 '
 '팔의 손목 이상을 잃었을 때 3) 한 팔의 3대 관절 중 관절 하나의 기능을 완전히 '
 '30</td><td>60</td></tr><tr><td>잃었을 때 4) 한 팔의 3대 관절 중 관절 하나의 기능에 심한 장해를 남긴 때 '
 '20</td><td>보'),
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
 'indexing': {'chunk_id': 'chunk_001576',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
