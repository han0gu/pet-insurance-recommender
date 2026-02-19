from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 회사가 계약 당시에 그 사실을 알았거나 과실로 인하여 알지 못하였을 때 2. 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또는 '
 '제1회 보험료 등을 받은 때부터 보험금 지급사유가 발생하지 않고 2년이 지났을 때 3. 계약을 체결한 날부터 3년이 지났을 때 4. '
 '보험을 모집한 자(이하 "보험설계사 등"이라 합니다)가 계약자 또는 피보험자에게 알릴 기회를 주지 않았거나 계약자 또는 피보험자가 '
 '사실대로 알리는 것을 방해한 경우, 계약자 또는 피보험 자에게 사실대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 권유했을 때'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000087',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
