from langchain_core.documents import Document

chunk = Document(
    page_content=('- 해\n'
 '- 할 수 있습니다.\n'
 '- 및\n'
 '- \uf000 제1항 및 제2항의 경우 회사는 계약자에게 이 특별약관의 해약환급금을 지급합니다.\n'
 '- 질\n'
 '# 제25조(특별약관의 소멸)질\uf000 보험증권에 기재된 반려동물이 보험기간 중에 사망하여 보험의 목적에 대해 이- 특별약관에서 정한 '
 '보험금 지급사유가 더 이상 발생할 수 없는 경우에는 이 특별\n'
 '- 반\n'
 '- 약관 계약도 소멸되며 회사는 "보험료 및 해약환급금 산출방법서"에서 정하는 바\n'
 '- 에 따라 피보험자의 사망 당시 이 특별약관의 계약자적립액 및 미경과보험료를 려동\n'
 '- 계약자에게 지급합니다. 물'),
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
 'indexing': {'chunk_id': 'chunk_000714',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
