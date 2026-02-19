from langchain_core.documents import Document

chunk = Document(
    page_content=('제4관 보험계약의 성립과 유지\n'
 '제19조(보험계약의 성립)\n'
 '\uf000 계약은 계약자의 청약과 회사의 승낙으로 이루어집니다. \uf000 회사는 피보험자가 계약에 적합하지 않은 경우에는 승낙 을 '
 '거절하거나 별도의 조건(보험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할 수 있 습니다.\n'
 '【 보험가입금액 제한 】\n'
 '피보험자가 가입을 할 수 있는 최대 보험가입금액을 제 한하는 방법을 말합니다.\n'
 '【 일부보장 제외(부담보) 】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 63},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000075',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
