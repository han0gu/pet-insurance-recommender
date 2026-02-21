from langchain_core.documents import Document

chunk = Document(
    page_content=('보험계약일 보장개시일(책임개시일)\n'
 '◄───── 30일주2) ─────►\n'
 '2022년 8월 1일 2022년 8월 31일주1) 상해를 직접적인 원인으로 치료를 받은 경우에는 보장개시일(책임개시일)은 보험계약일로 '
 '합\n'
 '니다.\n'
 '주2) 암, 백내장, 녹내장, 심장질환, 신장질환, 방광질환 및 각종 결석의 경우 90일<유의사항># [수술]동물병원의 수의사 자격을 '
 "가진 자(이하 '수의사'라 합니다)에 의하여 치료가 필요하다고 인정된 상"),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'eye', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000587',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
