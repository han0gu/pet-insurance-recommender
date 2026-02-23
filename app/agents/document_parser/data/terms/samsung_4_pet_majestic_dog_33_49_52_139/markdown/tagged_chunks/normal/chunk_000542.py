from langchain_core.documents import Document

chunk = Document(
    page_content=('또는 이에 준하는 전자적 의사표시 등으로 알려드리고, 회사는 계약자의 재가입의사\n'
 '를 전화(음성녹음), 직접 방문 또는 전자적 의사표시, 통신판매계약의 경우 통신수단\n'
 '을 통해 확인합니다.\n'
 '④ 계약자는 제3항에 따른 재가입안내와 재가입여부 확인 요청을 받은 경우 재가입 의사\n'
 '를 표시하여야 합니다.\n'
 '⑤ 제3항 및 제4항에도 불구하고, 회사가 계약자의 재가입 의사를 확인하지 못한 경우(계\n'
 '약자와의 연락두절로 회사의 안내가 계약자에게 도달하지 못한 경우 포함)에는 직전'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000542',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
