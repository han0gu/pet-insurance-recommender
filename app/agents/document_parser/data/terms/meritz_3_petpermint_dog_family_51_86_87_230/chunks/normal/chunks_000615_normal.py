from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사가 최초계약 체결당시에 그 사실을 알았거나 과실 로 알지 못하였을 때 ② 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또 는 '
 '제1회 보험료를 받은 때부터 보험금 지급사유가 발 생하지 않고 2년이 지났을 때 ③ 최초계약을 체결한 날부터 3년이 지났을 때 ④ '
 '보험설계사가 계약자 또는 피보험자에게 알릴 기회를 주지 않았거나 계약자 또는 피보험자가 사실대로 알리 는 것을 방해한 경우, 계약자 또는 '
 '피보험자에게 사실 대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 권유했을 때'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 182},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000615',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
