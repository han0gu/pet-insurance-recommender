from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 회사는 피보험자가 계약에 적합하지 않은 경우에는 승낙을 거절하거나 별도의 조건(보\n'
 '- 험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할 수 있습\n'
 '- 니다.\n'
 '- ③ 회사는 계약의 청약을 받고, 제1회 보험료를 받은 경우에 건강진단을 받지 않는 계약은\n'
 '- 청약일, 진단계약은 진단일(재진단의 경우에는 최종 진단일)부터 30일 이내에 승낙 또\n'
 '- 는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니다. 그러나 30일 이내에 승낙 또\n'
 '- 는 거절의 통지가 없으면 승낙된 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000067',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
