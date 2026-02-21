from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만,<br>상<br>지급사유의 조사나 확인이 필요한 경우 접수 후 10일 이내에 드립니다. 해</p><br><p id='230' "
 "data-category='list' style='font-size:14px'>\uf000 제1항에 따라 지급사유의 조사나 확인이 필요한 "
 '경우 계약자가 회사로부터의 사실 및<br>조회에 대하여 정당한 사유없이 회답 또는 동의를 거부한 때에는, 그 회답 또는 질<br>동의를 '
 '얻어 사실확인이 끝날 때까지 이 특별약관의 보험금을 지급하지 않습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001352',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
