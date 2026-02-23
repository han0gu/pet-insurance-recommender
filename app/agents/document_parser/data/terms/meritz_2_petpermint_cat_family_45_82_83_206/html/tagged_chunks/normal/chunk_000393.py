from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 때 회사는 해지 전 발생한 보험금 지급사유를 이유<br>로 부활(효력회복)을 거절하지 않습니다.</p><footer id='44' "
 "style='font-size:14px'>101</footer><p id='45' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제1항에서 정한 계약의 부활이 이루어진 경우라도 계약<br>자 또는 피보험자가 "
 '최초계약 청약시(2회 이상 부활이 이루<br>어진 경우 종전 모든 부활 청약 포함) 제7조(계약 전 알릴<br>의무)를 위반한 경우에는 '
 '제9조(알릴'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000393',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
