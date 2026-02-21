from langchain_core.documents import Document

chunk = Document(
    page_content=('- 안, 의치 등 신체보조장구는 제외하나, 인공장기나 부분 의치 등 신체에 이식되어\n'
 '- 그 기능을 대신할 경우는 포함합니다)에 입은 상해를 말합니다.\n'
 '- 2. 장해: [별표2] 장해분류표에서 정한 기준에 따른 장해상태를 말합니다.\n'
 '- 3. 중요한 사항: 계약 전 알릴 의무와 관련하여 회사가 그 사실을 알았더라면 계약의\n'
 '- 청약을 거절하거나 보험가입금액 한도 제한, 일부 보장 제외, 보험금 삭감, 보험료\n'
 '- 할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합\n'
 '- 니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000002',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
