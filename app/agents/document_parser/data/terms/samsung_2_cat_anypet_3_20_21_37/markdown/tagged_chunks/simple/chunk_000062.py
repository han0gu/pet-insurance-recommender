from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의한 계약), 제15조(보험계약의 성립), 제21조(제1회 보험료 등 및 회사의 보장개시) 및 제26조(계\n'
 '- 약의 해지)의 규정을 준용합니다. 이 때 회사는 해지 전 발생한 보험금 지급사유를 이유로 부활(효\n'
 '- 력회복)을 거절하지 않습니다.\n'
 '- ③ 제1항에서 정한 계약의 부활이 이루어진 경우라도 계약자 또는 피보험자가 최초 계약 청약시(2회\n'
 '- 이상 부활이 이루어진 경우 종전 모든 부활 청약 포함) 제12조(계약 전 알릴의무)를 위반한 경우에\n'
 '- 는 제26조(계약의 해지) 제3항이 적용됩니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
