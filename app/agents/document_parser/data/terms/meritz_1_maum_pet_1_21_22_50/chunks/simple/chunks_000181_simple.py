from langchain_core.documents import Document

chunk = Document(
    page_content=('⑤ 손해가 제1항 제1호 또는 제2호에 해당되는 사실로 생긴 것이 아님을 계약자 또는 피 보험자가 증명한 경우에는 제4항에 관계없이 '
 '보상합니다. ⑥ 회사는 다른 보험가입내역에 대한 계약 전․후 알릴 의무 위반을 이유로 계약을 해지하 거나 보험금 지급을 거절하지 '
 '않습니다. ⑦ 보통약관 제28조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))에 따라 이 계약이 부활이 이루어진 경우에는 '
 '부활계약을 제2항의 최초계약으로 봅니다.(부활(효력 회복)이 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 최초계약으로 봅니다)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 29},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
