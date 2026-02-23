from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또는 제1회 보험료를 받은 때부\n'
 '터 보험금 지급사유가 발생하지 않고 2년이 지났을 때\n'
 '3. 최초계약을 체결한 날부터 3년이 지났을 때\n'
 '4. 보험을 모집한 자(이하 “보험설계사 등”이라 합니다)가 계약자 또는 피보험자에게\n'
 '알릴 기회를 주지 않았거나 계약자 또는 피보험자가 사실대로 알리는 것을 방해한\n'
 '경우, 계약자 또는 피보험자에게 사실대로 알리지 않게 하였거나 부실한 사항을 알\n'
 '릴 것을 권유했을 때. 다만, 보험설계사 등의 행위가 없었다 하더라도 계약자 또는'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000151',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
