from langchain_core.documents import Document

chunk = Document(
    page_content=('- 출석에 협조하여야 합니다.\n'
 '- ③ 피보험자가 피해자로부터 손해배상의 청구를 받았을 경우에 회사가 필요하다고 인정\n'
 '- 할 때에는 피보험자를 대신하여 회사의 비용으로 이를 해결할 수 있습니다. 이 경우\n'
 '- 회사의 요구가 있으면 계약자 또는 피보험자는 이에 협력하여야 합니다.\n'
 '- ④ 계약자 및 피보험자가 정당한 이유 없이 제2항 및 제3항의 요구에 협조하지 않았을\n'
 '- 때에는 회사는 그로 인하여 늘어난 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000655',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
