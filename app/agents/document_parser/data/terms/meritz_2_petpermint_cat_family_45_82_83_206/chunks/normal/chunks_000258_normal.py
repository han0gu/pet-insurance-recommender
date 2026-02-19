from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제6항에 따라 보험계약이 연장된 경우 계약자는 회사에 \uf000 재가입 의사를 표시할 수 있습니다. 회사는 계약자의 재가 '
 '입 의사가 확인되었을 때에는 제2항 및 제3항에서 정한 절 차에 따라 회사가 재가입 의사를 확인한 날에 판매중인 제3 항의 반려동물보험 '
 '상품으로 재가입하는 것으로 하며, 기존 계약은 해지됩니다. 다만, 계약자가 재가입을 원하지 않는 경우에는 해당 시점으로부터 계약은 '
 '해지됩니다(단, 최초연 장된 날로부터 90일 이전에는 계약을 취소 또는 해지할 수 있습니다.)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 99},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000258',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
