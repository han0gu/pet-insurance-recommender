from langchain_core.documents import Document

chunk = Document(
    page_content=('진단확정)에서 정한 "2대호흡계특정질환"을 직접적인 원인으로 사망한 사실이\n'
 '확인된 경우에는 그 사망일을 진단 확정일로 보고 제1조(보험금의 지급사유)에\n'
 '해당하는 경우에 한하여 해당 보험금을 지급합니다. 다만, 제4조(특별약관의 소\n'
 '멸) 제2항에 따라 이 특별약관의 계약자적립액 등을 지급한 경우에는, 이미 지급\n'
 '된 계약자적립액 등을 차감하고 그 차액을 지급합니다.-'),
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
 'indexing': {'chunk_id': 'chunk_000386',
              'chunk_char_len': 204,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
