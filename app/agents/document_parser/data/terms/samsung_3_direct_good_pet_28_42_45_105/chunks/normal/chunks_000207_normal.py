from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑦ 회사는 다른 보험가입내역에 대한 계약 전 알릴 의무 위반을 이유로 특별약관을 해지 하거나 보험금 지급 및 보험료 납입면제를 '
 '거절하지 않습니다. ⑧ 제30조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))에 따라 이 특별 약관이 부활(효력회복)된 '
 '경우에는 부활(효력회복)계약을 제2항의 최초계약으로 봅니 다. 부활(효력회복)이 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 '
 '최초계 약으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 50},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000207',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
