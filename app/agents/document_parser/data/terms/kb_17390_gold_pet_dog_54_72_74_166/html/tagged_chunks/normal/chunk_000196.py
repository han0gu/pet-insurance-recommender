from langchain_core.documents import Document

chunk = Document(
    page_content=('변경하여 드립니다.<br>\uf000 회사는 계약자가 제1항 제5호에 따라 보험가입금액 또는 보상한도액을 감액하고자<br>할 때에는 그 '
 '감액된 부분은 해지된 것으로 보며, 이로써 회사가 지급하여야 할 해<br>약환급금이 있을 때에는 제34조(해약환급금) 제1항에 따른 '
 '해약환급금을 계약자에<br>게 지급합니다.<br>\uf000 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경우 계약자와 피보험자가 '
 '동<br>일하지 않을 때에는 보험금 지급사유가 발생하기 전에 피보험자가 서면(「전자서<br>명법」 제2조 제2호에 따른 전자서명이 있는 '
 '경우로서 상법'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000196',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
