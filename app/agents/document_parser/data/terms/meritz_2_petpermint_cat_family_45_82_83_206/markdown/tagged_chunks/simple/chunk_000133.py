from langchain_core.documents import Document

chunk = Document(
    page_content=('안내자료의 내용이 약관의 내용과 다른 경우에는 계약자에\n'
 '게 유리한 내용으로 계약이 성립된 것으로 봅니다.# 【보험안내자료】계약의 청약을 권유하기 위해 만든 자료 등을 말합니다.# 제44조(법령 '
 '등의 개정에 따른 계약내용의 변경)\uf000 회사는 보험금 지급사유 관련 법률이 개정된 경우에는\n'
 '변경된 내용을 적용합니다.\n'
 '\uf000 제1항에도 불구하고 다음 각 호 중 어느 한 가지에 해당\n'
 '되는 경우에는 회사는 객관적이고 합리적인 범위내에서 기\n'
 '존 계약내용에 상응하는 새로운 보장내용으로 계약내용을'),
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
 'indexing': {'chunk_id': 'chunk_000133',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
