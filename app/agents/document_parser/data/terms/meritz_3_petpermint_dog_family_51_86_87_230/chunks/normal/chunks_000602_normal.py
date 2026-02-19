from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 피보험자가 피해자로부터 손해배상의 청구를 받았을 경 우에 회사가 필요하다고 인정할 때에는 피보험자를 대신하 여 회사의 '
 '비용으로 이를 해결할 수 있습니다. 이 경우 회 사의 요구가 있으면 계약자 및 피보험자는 이에 협력하여야 합니다. \uf000 계약자 및 '
 '피보험자가 정당한 이유없이 제2항 및 제3항 의 요구에 협조하지 않은 때에는 회사는 그로 인하여 늘어 난 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 179},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000602',
              'chunk_char_len': 215,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
