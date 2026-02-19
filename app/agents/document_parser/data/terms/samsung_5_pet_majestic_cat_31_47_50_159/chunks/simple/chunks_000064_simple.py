from langchain_core.documents import Document

chunk = Document(
    page_content=('[위험변경에 따른 계약변경 절차]\n'
 '<유의사항>\n'
 '위험변경사항 통지(우편, 전화, 방문 등)\n'
 '↓\n'
 '계약자, 피보험자의 계약변경사항 확인 후 청약\n'
 '계약변경사항 인수 심사\n'
 '↓\n'
 '정산금액 처리(환급 또는 추가납입)\n'
 '계약변경 완료'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 37},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000064',
              'chunk_char_len': 120,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
