from langchain_core.documents import Document

chunk = Document(
    page_content=('[위험변경에 따른 계약변경 절차]\n'
 '위험변경사항 통지\n'
 '(우편, 전화, 방문 등)\n'
 '↓\n'
 '계약자,피보험자의 계약변경사항 확인 후 청약\n'
 '↓\n'
 '계약변경사항 인수 심사\n'
 '↓\n'
 '정산금액 처리\n'
 '(환급 또는 추가납입)\n'
 '↓\n'
 '계약변경 완료- ③ 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소된 경우에는 납입보험료를 감액\n'
 '- 하고, 이후 기간 보장을 위한 재원인 해약환급금 등의 차이로 인하여 발생한 정산금액(이\n'
 '- 하 “정산금액”이라 합니다)을 환급하여 드립니다. 한편 위험이 증가된 경우에는 납입보험'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000055',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
