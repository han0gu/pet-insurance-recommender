from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- | --- | --- |\n'
 '| 구분 | 구분 | 입/통원 중 수술을 하지 않은 날의 경우 | 입/통원 중 수술을 한 날의 경우 | 질 연간 총 병 보상한도액 |\n'
 '| 실속형 | 입원 | 1일당 10만원 한도 1일당 | 150만원 한도 | 1,000만원 |\n'
 '| 실속형 | 통원 | 1일당 10만원 한도 1일당 | 150만원 한도 | 1,000만원 상 |\n'
 '| 기본형Ⅰ | 입원 | 1일당 15만원 한도 1일당 | 200만원 한도 | 1,000만원 해 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000545',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
