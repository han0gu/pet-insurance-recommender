from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는 계약자의 재가<br>입 의사가 확인되었을 때에는 제2항 및 제3항에서 정한 절<br>차에 따라 회사가 재가입 의사를 확인한 '
 '날에 판매중인 제3<br>항의 반려동물보험 상품으로 재가입하는 것으로 하며, 기존<br>계약은 해지됩니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000373',
              'chunk_char_len': 133,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
