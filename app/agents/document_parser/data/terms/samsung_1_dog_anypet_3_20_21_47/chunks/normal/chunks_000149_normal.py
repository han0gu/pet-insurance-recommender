from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 피보험자가 피해자로부터 손해배상의 청구를 받았을 경우에 회사가 필요하다고 인정할 때에는 피 보험자를 대신하여 회사의 비용으로 이를 '
 '해결할 수 있습니다. 이 경우에 회사의 요구가 있으면 계 약자 또는 피보험자는 이에 협력하여야 합니다. ④ 계약자 및 피보험자가 정당한 '
 '이유 없이 제2항, 제3항의 요구에 협조하지 않았을 때에는 회사는 그 로 인하여 늘어난 손해는 보상하지 않습니다.\n'
 '제9조(합의. 절충. 중재. 소송의 협조. 대행 등)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000149',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
