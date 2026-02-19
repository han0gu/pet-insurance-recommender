from langchain_core.documents import Document

chunk = Document(
    page_content=('※ 약관에서 인용된 법·규정은「별표 및 참고」의 「약관에서 인용된 법·규정」에서 확인할 수 있습니다.\n'
 '특별약관 일반사항\n'
 '제1관 목적 및 용어의 정의\n'
 '제 1조 (목적)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 50},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000175',
              'chunk_char_len': 92,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
