from langchain_core.documents import Document

chunk = Document(
    page_content=('제 6조 (갱신계약의 보장내용 변경시 계약자 안내에 관한 사항)\n'
 '제3조(갱신계약의 보험계약 적용 특칙) 제1호의 법령 및 표준약관의 제·개정 또는 금융위 원회의 명령에 따른 약관 개정으로 갱신계약의 '
 '보장내용이 변경되는 경우, 회사는 제2조 제5항에도 불구하고 다음 각 호에 따라 계약자에게 안내합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 123},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000791',
              'chunk_char_len': 168,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
