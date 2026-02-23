from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 제4조(특별약관의 소멸) 제2항에</p><br><p id='209' data-category='paragraph' "
 "style='font-size:14px'>따라 이 특별약관의 계약자적립액 등을 지급한 경우에는, 이미 지급된 계약자적</p><br><p "
 "id='210' data-category='paragraph' style='font-size:14px'>지급합니다.</p><br><h1 "
 "id='211' style='font-size:14px'>립액 등을 차감하고 그 차액을</h1><br><p id='212'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000660',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
