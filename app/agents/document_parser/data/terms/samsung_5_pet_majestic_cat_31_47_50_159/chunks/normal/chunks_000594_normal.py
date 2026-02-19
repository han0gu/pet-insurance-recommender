from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 반려묘가 이 특별약관에 적합하지 않은 경우에는 승낙을 거절하거나 별도의 조건(보험가입금액 제한, 일부보장 제외, 보험금 삭감, '
 '보험료 할증 등)을 붙여 승낙할 수 있습니다. ③ 회사는 이 특별약관의 청약을 받고 제1회 보험료를 받은 경우에 건강진단을 받지 않 는 '
 '계약은 청약일, 진단계약은 진단일(재진단의 경우에는 최종진단일)부터 30일 이내 에 승낙 또는 거절하여야 하며, 승낙한 때에는 보험증권을 '
 '드립니다. 그러나 30일 이 내에 승낙 또는 거절의 통지가 없으면 승낙된 것으로 봅니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000594',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
