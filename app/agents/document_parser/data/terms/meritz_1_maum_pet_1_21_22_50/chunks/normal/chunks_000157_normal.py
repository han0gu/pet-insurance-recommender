from langchain_core.documents import Document

chunk = Document(
    page_content=('다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액\n'
 '【사례】\n'
 '※ 보상책임액의 합계액이 손해액을 초과하는 경우 : 계약A: 보상책임액 1,000만원 / 계약B: 보상책임액 1,000만원 / 손해액 : '
 '1,000만원 → 계약A보험회사 : 500만원 지급 = 1,000만원 × 1,000만원 / (1,000만원 + 1,000 만원) → '
 '계약B보험회사 : 500만원 지급 = 1,000만원 × 1,000만원 / (1,000만원 + 1,000 만원)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 25},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000157',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
