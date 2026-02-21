from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 계약자가 제1항 제5호에 따라 보험가입금액 또는 보상한도액을 감액하고 도\n'
 '- 자 할 때에는 그 감액된 부분은 해지된 것으로 보며, 이로써 회사가 지급하여야 성\n'
 '- 할 해약환급금이 있을 때에는 보통약관 제1절 일반조항 제34조(해약환급금) 제1 특\n'
 '- 약\n'
 '- 항에 따른 해약환급금을 계약자에게 지급합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000505',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
