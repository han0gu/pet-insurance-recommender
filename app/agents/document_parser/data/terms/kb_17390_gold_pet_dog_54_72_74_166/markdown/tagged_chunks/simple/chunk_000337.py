from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 보통약관 제1절 일반조항 제5조(보험금을 지급하지 않는 사유) 및 다음 중 어\n'
 '느 한 가지의 경우로 인하여 보험금 지급사유가 발생한 때에는 보험금을 지급하지 않- 습니다.\n'
 '- 1. 위생관리, 미모를 위한 성형수술(다만, 사고전 상태로의 회복을 위한 수술은\n'
 '- 보장합니다)\n'
 '- 2. 선천적 기형 및 이에 근거한 병상\n'
 '# 제6조(수술의 정의와장소)- \uf000 이 특별약관에 있어서 "수술"이라 함은 병원 또는 의원의 의사, 치과의사 면허를\n'
 '- 가진 자(이하 "의사"라 합니다)에 의하여 치료가 필요하다고 인정한 경우로서 자'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000337',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
