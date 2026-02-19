from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 아래의 사유를 원인으로 하여 생긴 손해는 보상하지 않습니다.\n'
 '1. 보통약관 제5조 (보험금을 지급하지 않는 사유) 제1항 2. 피보험자의 치매를 제외한 정신적 기능장해, 선천성 뇌질환 및 심신상실 '
 '3. 성병 4. 알콜중독, 습관성 약품 또는 환각제의 복용 및 사용\n'
 '② 회사는 아래의 사유로 생긴 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 127},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000802',
              'chunk_char_len': 185,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
