from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제<br>\uf000 회사는 계약자가 제1항 제5호에 따라 보험가입금액 또는 보상한도액을 감액하고 도<br>자 할 때에는 그 감액된 '
 '부분은 해지된 것으로 보며, 이로써 회사가 지급하여야 성<br>할 해약환급금이 있을 때에는 보통약관 제1절 일반조항 제34조(해약환급금) '
 "제1 특<br>약<br>항에 따른 해약환급금을 계약자에게 지급합니다.</p><br><p id='34' "
 "data-category='paragraph' style='font-size:14px'>반</p><p id='35' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000863',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
