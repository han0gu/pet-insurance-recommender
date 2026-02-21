from langchain_core.documents import Document

chunk = Document(
    page_content=('→ 계약A보험회사 : 500만원 지급 = 1,000만원 × 1,000만원 / (1,000만원 + 1,000\n'
 '만원)\n'
 '→ 계약B보험회사 : 500만원 지급 = 1,000만원 × 1,000만원 / (1,000만원 + 1,000\n'
 '만원)② 이 특별약관이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 의무보험에서'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000135',
              'chunk_char_len': 173,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
