from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 때 사망을 보험금 지급사<br>통<br>유로 하지 않는 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 '
 '따라<br>회사가 적립한 사망 당시의 계약자적립액 및 미경과보험료(적립한 금액에서 중도인출 사항<br>액이 있었던 경우에는 그 원금과 '
 "이자의 합계액을 차감한 후의 금액)를 계약자에게</p><br><table id='20' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>지급합니다.</td><td></td></tr><tr><td "
 'colspan="2">부 가 설 명'),
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
 'indexing': {'chunk_id': 'chunk_000209',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
