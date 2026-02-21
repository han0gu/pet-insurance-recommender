from langchain_core.documents import Document

chunk = Document(
    page_content=('은 후 이를 숨기고 가입하는 등 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경\n'
 '우에는 계약일부터 5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습\n'
 '니다.제4관 보험계약의 성립과 유지# 제 20조 (보험계약의 성립)- ① 계약은 계약자의 청약과 회사의 승낙으로 이루어집니다.\n'
 '- ② 회사는 피보험자가 계약에 적합하지 않은 경우에는 승낙을 거절하거나 별도의 조건(보\n'
 '- 험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할 수 있\n'
 '- 습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000072',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
