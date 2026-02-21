from langchain_core.documents import Document

chunk = Document(
    page_content=('- 자 또는 피보험자의 고의 또는 중대한 과실로 이행하\n'
 '- 지 않았을 때\n'
 '\uf000 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당\n'
 '하는 경우에는 회사는 계약을 해지할 수 없습니다.- ① 회사가 최초계약 체결당시에 그 사실을 알았거나 과실\n'
 '- 로 알지 못하였을 때\n'
 '- ② 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또\n'
 '- 는 제1회 보험료를 받은 때부터 보험금 지급사유가 발\n'
 '- 생하지 않고 2년(진단계약의 경우 질병에 대하여는 1\n'
 '- 년)이 지났을 때\n'
 '- ③ 최초계약을 체결한 날부터 3년이 지났을 때'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000055',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
