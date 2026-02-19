from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 반려동물이 이 특별약관에서 정한 회사가 보험금을 지급 하지 않는 기간의 종료일을 포함하여 계속하여 입원한 경우 그 입원에 '
 '대해서는 회사가 보험금을 지급하지 않는 기간 종료일의 다음날을 입원의 개시일로 인정하여 보험금을 지 급합니다. \uf000 반려동물에게 '
 '보험금의 지급사유가 발생했을 경우, 그 보험금의 지급사유가 특정질병을 직접적인 원인으로 발생한 보험금의 지급사유인지 아닌지는 수의사의 '
 '진단서와 의견을 주된 판단자료로 하여 결정합니다.\n'
 '제3조(특별약관의 부활(효력회복))\n'
 '회사는 이 특별약관의 부활(효력회복) 청약을 받은 경우에'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 167},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000585',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
