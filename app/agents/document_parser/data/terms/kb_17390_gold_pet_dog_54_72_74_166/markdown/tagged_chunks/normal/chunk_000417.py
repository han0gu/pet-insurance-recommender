from langchain_core.documents import Document

chunk = Document(
    page_content=('- 끝났을 때에도 퇴원하기 전까지의 계속중인 입원기간에 대하여는 제1조(보험금의\n'
 '- 지급사유) 제2항에 따라 환경성질환입원일당을 계속 지급합니다.\n'
 '- \uf000 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회\n'
 '- 사는 환경성질환입원일당의 전부 또는 일부를 지급하지 않습니다. 상\n'
 '- \uf000 피보험자가 병원 또는 의원을 이전하여 입원한 경우에도 동일한 질병의 치료를 해\n'
 '- 목적으로 2회 이상 입원한 경우에는 계속하여 입원한 것으로 보아 각 입원일수를 및\n'
 '- 질\n'
 '- 더합니다.\n'
 '- 병'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000417',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
