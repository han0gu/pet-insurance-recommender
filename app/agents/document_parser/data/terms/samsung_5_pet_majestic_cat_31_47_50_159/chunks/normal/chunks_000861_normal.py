from langchain_core.documents import Document

chunk = Document(
    page_content=(". 6. 회사가 해지권을 행사하는 경우 위 표의 '청구일' 은 회사의 해지 의사표시(서면, 전자우편, 휴대전화 문자메시지 또는 이에 "
 '준하는 전자적 의사표시 포함)가 보험계약자 또는 그의 대리 인에게 도달한 날로 봅니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 135},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000861',
              'chunk_char_len': 122,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
