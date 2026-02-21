from langchain_core.documents import Document

chunk = Document(
    page_content=('|  |\n'
 '| --- |\n'
 '| 예 시 중도인출금의 한도 중도인출 시점에 "보험료 및 해약환급금 산출방법서"에 의해 산출된 기본계약 공 해약환급금과 적립부분 '
 '해약환급금 중 적은 금액이 100만원인 경우 통 ⇒ 총 중도인출 가능액 = 100만원× 80% = 80만원 사항 ⇒ 기 신청한 대출금이 '
 '있는 경우(원금과 이자의 합계를 10만원으로 가정) |'),
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
 'indexing': {'chunk_id': 'chunk_000191',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
