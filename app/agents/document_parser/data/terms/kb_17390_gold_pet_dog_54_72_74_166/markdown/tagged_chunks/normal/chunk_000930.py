from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 장해의 분류 지급률 | 사항 |\n'
 '| 1) 신경계에 장해가 남아 일상생활 기본동작에 제한을 남긴 때 | 10~100 |\n'
 '| 2) 정신행동에 극심한 장해를 남긴때 | 100 |\n'
 '| 3) 정신행동에 심한 장해를 남긴 때 | 75 보 |\n'
 '| 4) 정신행동에 뚜렷한 장해를 남긴 때 | 50 통약 |\n'
 '| 5) 정신행동에 약간의 장해를 남긴 때 | 25 관 |\n'
 '| 6) 정신행동에 경미한 장해를 남긴 때 | 10 |\n'
 '| 7) 극심한 치매 : CDR척도 5점 | 100 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000930',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
