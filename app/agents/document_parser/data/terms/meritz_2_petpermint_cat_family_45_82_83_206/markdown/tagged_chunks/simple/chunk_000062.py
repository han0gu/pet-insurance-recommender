from langchain_core.documents import Document

chunk = Document(
    page_content=('립되었음을 회사가 증명하는 경우에는 계약일부터 5년 이내\n'
 '(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있\n'
 '습니다.\n'
 '\uf000 제1항에 따라 계약이 취소된 경우에는 회사는 이미 납입\n'
 '한 보험료를 계약자에게 돌려 드립니다.62# 제4관 보험계약의 성립과 유지# 제19조(보험계약의 성립)\uf000 계약은 계약자의 청약과 '
 '회사의 승낙으로 이루어집니다.\n'
 '\uf000 회사는 피보험자가 계약에 적합하지 않은 경우에는 승낙\n'
 '을 거절하거나 별도의 조건(보험가입금액 제한, 일부보장\n'
 '제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할 수 있'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
