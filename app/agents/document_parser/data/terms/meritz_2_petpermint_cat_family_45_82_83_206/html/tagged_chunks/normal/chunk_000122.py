from langchain_core.documents import Document

chunk = Document(
    page_content=('청약일로부터 5년(갱신형 계약의 경우에는 최초<br>계약의 청약일 이후 5년)이 지나는 동안 보장이 제외되는<br>질병으로 추가 '
 '진단(단순 건강검진 제외) 또는 치료사실이<br>없을 경우, 청약일로부터 5년이 지난 이후에는 이 약관에<br>따라 '
 '보장합니다.<br>\uf000 제5항의「청약일로부터 5년이 지나는 동안」이라 함은<br>제29조(보험료의 납입이 연체되는 경우 '
 '납입최고(독촉)와<br>계약의 해지)에서 정한 계약의 해지가 발생하지 않은 경우<br>를 말합니다.<br>\uf000 제30조(보험료의 '
 '납입을 연체하여 해지된 계약의'),
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
 'indexing': {'chunk_id': 'chunk_000122',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
