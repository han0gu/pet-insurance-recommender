from langchain_core.documents import Document

chunk = Document(
    page_content=('제24조(계약의 소멸)\n'
 '피보험자의 사망으로 인하여 이 약관에서 규정하는 보험금 지급사유가 더 이상 발생할\n'
 '공\n'
 '수 없는 경우에는 이 계약은 그 때부터 효력이 없습니다. 이 때 사망을 보험금 지급사\n'
 '통\n'
 '유로 하지 않는 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라\n'
 '회사가 적립한 사망 당시의 계약자적립액 및 미경과보험료(적립한 금액에서 중도인출 사항\n'
 '액이 있었던 경우에는 그 원금과 이자의 합계액을 차감한 후의 금액)를 계약자에게| 지급합니다. |  |\n'
 '| --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000136',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
