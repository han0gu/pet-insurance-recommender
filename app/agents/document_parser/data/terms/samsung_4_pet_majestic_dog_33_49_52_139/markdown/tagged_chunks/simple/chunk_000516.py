from langchain_core.documents import Document

chunk = Document(
    page_content=('습니다.\n'
 '② 보험의 목적이 다수인 경우 제1항은 보험의 목적별로 각각 적용합니다.- \n'
 '# 제20조 (제1회 보험료 및 회사의 보장개시)① 회사는 특별약관의 청약을 승낙하고 제1회 보험료를 받은 때부터 이 약관이 정한 바\n'
 '에 따라 보장을 합니다. 또한, 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙\n'
 '한 경우에는 제1회 보험료를 받은 때부터 보장이 개시됩니다. 자동이체 또는 신용카\n'
 '드로 납입하는 경우에는 자동이체신청 또는 신용카드매출승인에 필요한 정보를 제공'),
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
 'indexing': {'chunk_id': 'chunk_000516',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
