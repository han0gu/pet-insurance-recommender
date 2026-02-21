from langchain_core.documents import Document

chunk = Document(
    page_content=('요청이 있는 경<br>우에 한하여 "보험료 및 해약환급금 산출방법서"에 따라 계약자가 요청한 시점에서<br>계산된 기본계약 해약환급금과 '
 '적립부분 해약환급금 중 적은 금액(적립한 금액에<br>서 이 계약에서 정한 대출금이 있을 때에는 그 원금과 이자의 합계액을 차감한 '
 '후<br>의 잔액을 기준으로 합니다)의 80% 범위 내에서 회사가 정한 방법에 따라 중도인출<br>을 할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000285',
              'chunk_char_len': 214,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
