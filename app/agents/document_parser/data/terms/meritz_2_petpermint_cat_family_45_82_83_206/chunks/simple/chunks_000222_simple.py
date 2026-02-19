from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사가 최초계약 체결당시에 그 사실을 알았거나 과실 로 인하여 알지 못하였을 때 ② 회사가 그 사실을 안 날부터 1개월 이상 지났거나 '
 '또 는 제1회 보험료를 받은 때부터 보험금 지급사유가 발 생하지 않고 2년(진단계약의 경우 질병에 대하여는 1 년)이 지났을 때 ③ '
 '최초계약을 체결한 날부터 3년이 지났을 때 ④ 회사가 이 계약을 청약할 때 반려동물의 건강상태를 판단할 수 있는 기초자료(건강진단서 사본 '
 '등)에 따라 승낙한 경우에 건강진단서 사본 등에 명기되어 있는 사항으로 보험금 지급사유가 발생하였을 때(계약자 또 는 피보험자가'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000222',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
