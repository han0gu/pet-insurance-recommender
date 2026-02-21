from langchain_core.documents import Document

chunk = Document(
    page_content=('. 병<br>또한, 회사가 지정한 의사에 의한 피보험자의 진단을 요구한 경우에는 진단을 받<br>지 않는 때에는 진단을 받고 사실 확인이 '
 '끝날 때까지 이 특별약관의 보험금을 지<br>급하지 않습니다'),
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
 'indexing': {'chunk_id': 'chunk_001353',
              'chunk_char_len': 110,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
