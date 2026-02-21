from langchain_core.documents import Document

chunk = Document(
    page_content=('위험변경사항 통지(우편, 전화, 방문 등)\n'
 '↓\n'
 '계약자, 피보험자의 계약변경사항 확인 후 청약\n'
 '↓\n'
 '계약변경사항 인수 심사\n'
 '↓\n'
 '정산금액 처리(환급 또는 추가납입)\n'
 '↓\n'
 '계약변경 완료- ③ 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소된 경우에는 보험료를 감액하\n'
 '- 고, 이후 기간 보장을 위한 재원인 계약자적립액 등의 차이로 인하여 발생한 정산금액\n'
 '- (이하 「정산금액」이라 합니다)을 환급하여 드립니다. 한편 위험이 증가된 경우에는\n'
 '- 보험료의 증액 및 정산금액의 추가납입을 요구할 수 있으며, 계약자는 이를 납입하여'),
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
 'indexing': {'chunk_id': 'chunk_000496',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
