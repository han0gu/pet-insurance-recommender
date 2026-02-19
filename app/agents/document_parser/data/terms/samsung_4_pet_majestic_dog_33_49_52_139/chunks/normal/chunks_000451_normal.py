from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 상해 입원일당(1일이 상)의 지급일수는 1회 입원당 180일을 한도로 합니다. ② 제1항의 경우 피보험자가 동일한 상해의 '
 '치료를 직접적인 목적으로 2회 이상 입원한 경우 이를 1회 입원으로 보아 입원일수를 더합니다. ③ 제1항의 경우 피보험자가 병원 또는 '
 '의원을 이전하여 입원한 경우에도 동일한 상해의 치료를 직접적인 목적으로 입원한 경우에는 계속하여 입원한 것으로 보아 각 입원일 수를 '
 '더합니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 85},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000451',
              'chunk_char_len': 227,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
