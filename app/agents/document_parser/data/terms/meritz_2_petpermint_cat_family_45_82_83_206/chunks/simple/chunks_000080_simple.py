from langchain_core.documents import Document

chunk = Document(
    page_content=('하는 경우에는 신용카드의 매출을 취소하며 이자를 더하여 지급하지 않습니다. \uf000 회사가 제2항에 따라 일부보장 제외 조건을 붙여 '
 '승낙하 였더라도 청약일로부터 5년(갱신형 계약의 경우에는 최초 계약의 청약일 이후 5년)이 지나는 동안 보장이 제외되는 질병으로 추가 '
 '진단(단순 건강검진 제외) 또는 치료사실이 없을 경우, 청약일로부터 5년이 지난 이후에는 이 약관에 따라 보장합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 64},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000080',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
