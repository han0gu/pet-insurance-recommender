from langchain_core.documents import Document

chunk = Document(
    page_content=('| 7) 극심한 치매 : CDR척도 5점 | 100 |\n'
 '| 8) 심한치매 : CDR척도 4점 | 80 |\n'
 '| 9) 뚜렷한 치매 : CDR 척도 3점 | 60 특별 |\n'
 '| 10) 약간의 치매 : CDR 척도 2점 | 약 40 관 |\n'
 '| 11) 심한 뇌전증 발작이 남았을 때 | 70 |\n'
 '| 12) 뚜렷한 뇌전증 발작이 남았을 때 | 40 |\n'
 '| 13) 약간의 뇌전증 발작이 남았을 때 | 10 |\n'
 '- 나. 장해판정기준\n'
 '1)신경계# 가) “신경계에 장해를 남긴 때”라 함은 뇌, 척수 및 말초신경계 손상으법'),
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
 'indexing': {'chunk_id': 'chunk_000931',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
