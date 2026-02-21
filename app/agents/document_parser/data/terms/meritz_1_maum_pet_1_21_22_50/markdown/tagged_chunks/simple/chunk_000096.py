from langchain_core.documents import Document

chunk = Document(
    page_content=('- 제25조(제1회 보험료 및 회사의 보장개시)의 규정을 준용합니다. 이 때 회사는 해지 전\n'
 '- 발생한 보험금 지급사유를 이유로 부활(효력회복)을 거절하지 않습니다.\n'
 '- ③ 제1항에서 정한 계약의 부활이 이루어진 경우라도 계약자 또는 피보험자가 최초 계약\n'
 '- 청약시(2회 이상 부활이 이루어진 경우 종전 모든 부활 청약 포함) 제15조(계약 전 알\n'
 '- 릴 의무)를 위반한 경우에는 제17조(알릴 의무 위반의 효과)가 적용됩니다.\n'
 '# 【부활(효력회복)】보험료 납입을 연체하여 계약이 해지되고 계약자가 해약환급금을 받지 않은 경우'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000096',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
