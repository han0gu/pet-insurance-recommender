from langchain_core.documents import Document

chunk = Document(
    page_content=('. ㆍ<br>\uf000 회사는 계약의 청약을 받고, 제1회 보험료를 받은 경우에 건강진단을 받지 않는 계 규정<br>약은 청약일, '
 '진단계약은 진단일(재진단의 경우에는 최종 진단일)부터 30일 이내<br>에 승낙 또는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니다'),
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
 'indexing': {'chunk_id': 'chunk_000138',
              'chunk_char_len': 145,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
