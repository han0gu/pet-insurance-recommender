from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 보험증권에 기재된 반려견이 보험기간 중에 이 특별약관에서 보장하지 않는 사유로\n'
 '- 사망하였을 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 회사\n'
 '- 가 적립한 사망당시 이 특별약관의 계약자적립액 및 미경과보험료를 계약자에게 지급\n'
 '- 하고, 이 특별약관은 더 이상 효력이 없습니다.\n'
 '- ③ 보험의 목적이 다수인 경우 제1항 내지 제2항은 보험의 목적별로 각각 적용합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000629',
              'chunk_char_len': 221,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
