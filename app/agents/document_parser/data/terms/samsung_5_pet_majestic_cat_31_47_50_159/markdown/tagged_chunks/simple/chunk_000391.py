from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 사건사고사실확인원(관할 경찰서 발행), 법원의 판결문 또는 검찰의 공소장, 검찰청\n'
 '- 에서 발행한 불기소이유통지서 등(죄명 및 피의자와 피보험자와의 관계를 알 수 있\n'
 '- 는 서류)\n'
 '- 3. 사고증명서(진단서 또는 상해진단서, 진료비계산서, 사망진단서, 장해진단서, 입원\n'
 '- 치료확인서, 의사처방전(처방조제비) 등)\n'
 '- 4. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본인이\n'
 '- 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000391',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
