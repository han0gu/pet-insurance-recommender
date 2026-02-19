from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 회사는 계약자의 자동갱신 의사를 전화(음성녹음), 직접 방문 또는 전자적 의사표시 (통신판매계약의 경우 통신수단) 등을 통해 '
 '확인하고, 자동갱신 의사가 확인되는 경 우 갱신전 계약은 갱신일에 갱신일 현재의 약관 등으로 갱신됩니다. 다만, 계약자가 자동갱신을 '
 '원하지 않는 경우에는 갱신일에 갱신전 계약은 만료됩니다. 3. 회사가 계약자의 자동갱신 의사를 확인하지 못한 경우(계약자와 연락두절 '
 '등으로 회 사 안내가 계약자에게 도달하지 못한 경우 포함)에는 갱신일에 갱신일 현재의 약관 등으로 갱신됩니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 131},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000820',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
