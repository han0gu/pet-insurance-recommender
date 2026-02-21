from langchain_core.documents import Document

chunk = Document(
    page_content=(". 해</p><br><p id='108' data-category='list' style='font-size:16px'>약환급금 "
 '지급일까지의 기간에 대한 이자의 계산은 "보험금을 지급할 때의 적립이<br>율 계산"(【별표2】참조)에 따릅니다.<br>\uf000 '
 '계약자가 제36조(중도인출)에서 정한 방법에 따라 중도인출시 인출금액 및 해약환<br>급금의 지급 시점까지 인출금액에 적립되었을 이자만큼 '
 '해약환급금이 감소합니다.<br>\uf000 회사는 경과기간별 해약환급금에 관한 표를 계약자에게 제공하여 드립니다.<br>\uf000 '
 '제31조의1(위법계약의 해지)에'),
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
 'indexing': {'chunk_id': 'chunk_000280',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
