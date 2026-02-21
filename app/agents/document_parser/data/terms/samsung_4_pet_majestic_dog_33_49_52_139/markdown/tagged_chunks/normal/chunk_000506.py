from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약관이 부활(효력회복)된 경우에는 부활(효력회복)계약을 제2항의 최초계약으로 봅니\n'
 '- 다. 부활(효력회복)이 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 최초계\n'
 '- 약으로 봅니다.\n'
 '# 제 14조 (특별약관의 체결)- ① 이 특별약관은 기본계약을 체결할 때 계약자의 청약과 회사의 승낙으로 기본계약에\n'
 '- 부가하여 이루어집니다.\n'
 '- ② 회사는 반려견이 이 특별약관에 적합하지 않은 경우에는 승낙을 거절하거나 별도의\n'
 '- 조건(보험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할\n'
 '- 수 있습니다.'),
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
 'indexing': {'chunk_id': 'chunk_000506',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
