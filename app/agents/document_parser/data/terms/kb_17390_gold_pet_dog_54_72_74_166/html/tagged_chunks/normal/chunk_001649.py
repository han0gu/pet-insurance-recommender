from langchain_core.documents import Document

chunk = Document(
    page_content=('정신행동에 뚜렷한 장해를 남긴 때</td><td>50 통약</td></tr><tr><td>5) 정신행동에 약간의 장해를 남긴 '
 '때</td><td>25 관</td></tr><tr><td>6) 정신행동에 경미한 장해를 남긴 '
 '때</td><td>10</td></tr><tr><td>7) 극심한 치매 : CDR척도 '
 '5점</td><td>100</td></tr><tr><td>8) 심한치매 : CDR척도 '
 '4점</td><td>80</td></tr><tr><td>9) 뚜렷한 치매 : CDR 척도 3점</td><td>60'),
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
 'indexing': {'chunk_id': 'chunk_001649',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
