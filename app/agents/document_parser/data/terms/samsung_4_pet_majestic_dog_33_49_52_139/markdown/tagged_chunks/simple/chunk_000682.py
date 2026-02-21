from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하시고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다.\n'
 '- 4. 사고증명서(진단서, 진료비계산서, 사망진단서, 장해진단서, 입원치료확인서, 의사\n'
 '- 처방전(처방조제비) 등)\n'
 '- 5. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이\n'
 '- 아닌 경우에는 본인의 인감증명서 또는 안전성과 신뢰성이 확보된 전자적 수단을\n'
 '- 활용한 피보험자 의사표시의 확인방법 포함)\n'
 '- 6. 수탁기관 위탁비용 영수증 및 동물관리위탁업자가 제공하는 계약서(위탁관리업소'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000682',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
