from langchain_core.documents import Document

chunk = Document(
    page_content=('제 2조 (용어의 정의)\n'
 '이 특별약관에서 사용되는 용어의 정의는, 이 특별약관의 다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.\n'
 '① 계약관계 관련 용어'),
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
 'indexing': {'chunk_id': 'chunk_000177',
              'chunk_char_len': 88,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
