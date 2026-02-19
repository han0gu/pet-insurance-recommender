from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 회사의 요구가 있으면 계약자 또는 피보험자는 이에 협력하여야 합니다. ④ 계약자 및 피보험자가 정당한 이유 없이 제2항 및 '
 '제3항의 요구에 협조하지 않았을 때에는 회사는 그로 인하여 늘어난 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 89},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000578',
              'chunk_char_len': 127,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
