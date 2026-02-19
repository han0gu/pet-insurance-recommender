from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조 (갱신계약의 보장내용 변경시 계약자 안내에 관한 사항)\n'
 '제3조(갱신계약의 보험계약 적용 특칙) 제1호의 법령 및 표준약관의 제·개정 또는 금융위 원회의 명령에 따른 약관 개정으로 갱신계약의 '
 '보장내용이 변경되는 경우, 회사는 제2조 제5항에도 불구하고 다음 각 호에 따라 계약자에게 안내합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 130},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000817',
              'chunk_char_len': 167,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
