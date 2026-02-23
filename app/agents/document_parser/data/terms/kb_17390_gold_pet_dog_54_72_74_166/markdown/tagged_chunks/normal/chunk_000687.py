from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 증인출석에 협조하여야 합니다.\n'
 '- \uf000 피보험자가 피해자로부터 손해배상의 청구를 받았을 경우에 회사가 필요하다고\n'
 '- 인정할 때에는 피보험자를 대신하여 회사의 비용으로 이를 해결할 수 있습니\n'
 '- 다. 이 경우에 회사의 요구가 있으면 계약자 또는 피보험자는 이에 협력하여야\n'
 '- 합니다.\n'
 '- \uf000 계약자 또는 피보험자가 정당한 이유없이 제2항, 제3항의 요구에 협조하지 않았\n'
 '- 을 때에는 회사는 그로 인하여 늘어난 손해를 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000687',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
