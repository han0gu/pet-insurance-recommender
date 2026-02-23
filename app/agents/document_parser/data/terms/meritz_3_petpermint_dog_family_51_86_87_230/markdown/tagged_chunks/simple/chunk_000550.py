from langchain_core.documents import Document

chunk = Document(
    page_content=('하지 않는 기간의 종료일을 포함하여 계속하여 입원한 경우\n'
 '그 입원에 대해서는 회사가 보험금을 지급하지 않는 기간\n'
 '종료일의 다음날을 입원의 개시일로 인정하여 보험금을 지\n'
 '급합니다.\n'
 '\uf000 반려동물에게 보험금의 지급사유가 발생했을 경우, 그\n'
 '보험금의 지급사유가 특정질병을 직접적인 원인으로 발생한\n'
 '보험금의 지급사유인지 아닌지는 수의사의 진단서와 의견을\n'
 '주된 판단자료로 하여 결정합니다.# 제3조(특별약관의 부활(효력회복))회사는 이 특별약관의 부활(효력회복) 청약을 받은 경우에\n'
 '는 보험계약의 부활(효력회복)을 승낙한 경우에 한하여 보'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000550',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
