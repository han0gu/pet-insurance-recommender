from langchain_core.documents import Document

chunk = Document(
    page_content=('것이 아님을 계약자 또는 피보험자가 증명한 경우에는 제4\n'
 '항에 관계없이 보상합니다.\n'
 '\uf000 회사는 다른 보험가입내역에 대한 계약 전․후 알릴 의무\n'
 '위반을 이유로 계약을 해지하거나 보험금 지급을 거절하지\n'
 '않습니다.\n'
 '\uf000 반려동물 비용손해 관련 특별약관 일반조항 제18조(보험\n'
 '료의 납입을 연체하여 해지된 계약의 부활(효력회복))에 따\n'
 '라 이 계약이 부활이 이루어진 경우에는 부활계약을 제2항\n'
 '의 최초계약으로 봅니다.(부활(효력회복)이 여러차례 발생\n'
 '된 경우에는 각각의 부활(효력회복)계약을 최초계약으로 봅'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000513',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
