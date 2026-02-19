from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 보험금 청구서(회사양식) 2. 사건사고사실확인원(관할 경찰서 발행), 법원의 판결문 또는 검찰의 공소장, 검찰청 에서 발행한 '
 '불기소이유통지서 등(죄명 및 피의자와 피보험자와의 관계를 알 수 있 는 서류) 3. 사고증명서(진단서 또는 상해진단서, 진료비계산서, '
 '사망진단서, 장해진단서, 입원 치료확인서, 의사처방전(처방조제비) 등) 4'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 87},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000467',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
