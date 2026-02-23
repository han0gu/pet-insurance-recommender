from langchain_core.documents import Document

chunk = Document(
    page_content=('상# 제5조(특별약관의 소멸)해- \uf000 보험증권에 기재된 반려동물이 보험기간 중에 사망하여 보험의 목적에 대해 이\n'
 '- 질\n'
 '- 특별약관에서 정한 보험금 지급사유가 더 이상 발생할 수 없는 경우 회사는 "보 병\n'
 '- 험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 반려동물 사망 당시 이\n'
 '- 특별약관의 계약자적립액 및 미경과보험료를 계약자에게 지급합니다.\n'
 '- \uf000 보험의 목적이 다수인 경우 제1항은 보험의 목적별로 각각 적용합니다.\n'
 '- 반'),
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
 'indexing': {'chunk_id': 'chunk_000569',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
