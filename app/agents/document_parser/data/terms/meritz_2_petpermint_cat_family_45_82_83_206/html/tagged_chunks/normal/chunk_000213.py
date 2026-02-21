from langchain_core.documents import Document

chunk = Document(
    page_content=("임의해지 및 피보험자의 서면동의 철회)</p><br><p id='80' data-category='paragraph' "
 "style='font-size:16px'>\uf000 계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지<br>할 수 있으며, 이 경우 "
 '회사는 제35조(해약환급금) 제1항에<br>따른 해약환급금을 계약자에게 지급합니다.<br>\uf000 제22조(계약의 무효)에 따라 '
 '사망을 보험금 지급사유로<br>하는 계약에서 서면으로 동의를 한 피보험자는 계약의 효력<br>이 유지되는 기간에는 언제든지 서면동의를 '
 '장래를 향하여<br>철회할 수'),
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
 'indexing': {'chunk_id': 'chunk_000213',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
