from langchain_core.documents import Document

chunk = Document(
    page_content=("id='63' style='font-size:14px'>【사례】</h1><br><p id='64' "
 "data-category='paragraph' style='font-size:14px'>※ 보상책임액의 합계액이 손해액을 초과하는 경우 "
 ':<br>계약A: 보상책임액 1,000만원 / 계약B: 보상책임액 1,000만원 / 손해액 : 1,000만원<br>→ 계약A보험회사 : '
 '500만원 지급 = 1,000만원 × 1,000만원 / (1,000만원 + 1,000<br>만원)<br>→ 계약B보험회사 : 500만원 '
 '지급 = 1,000만원 ×'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000238',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
