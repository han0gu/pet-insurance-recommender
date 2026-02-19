from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항의 회사가 보험금을 지급하지 않는 기간(이하 「부 담보 기간」이라 합니다)은 특정질병의 상태에 따라「1개월 부터 '
 '5년」또는 「계약의 보험기간」(단, 계약이 갱신 또는 재가입 계약인 경우 최초 계약일로부터 최종 갱신 또는 재 가입 계약의 종료일까지의 '
 '기간을 말하며, 이하 「계약의 보험기간」이라 합니다)으로 하며, 그 판단기준은 회사에서 정한 계약사정기준을 따릅니다. 다만, 각각의 '
 '질병의 상태 등에 대한 수의사의 소견에 따라 다르게 적용할 수 있습니'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 166},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000580',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
