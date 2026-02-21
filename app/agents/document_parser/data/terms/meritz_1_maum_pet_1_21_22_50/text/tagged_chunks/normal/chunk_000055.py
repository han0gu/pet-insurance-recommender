from langchain_core.documents import Document

chunk = Document(
    page_content=('해지할 수 없습니다.1. 회사가 최초계약 체결당시에 그 사실을 알았거나 과실로 인하여 알지 못하였을 때\n'
 '2. 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또는 제1회 보험료를 받은 때부\n'
 '터 보험금 지급사유가 발생하지 않고 2년(진단계약의 경우 질병에 대하여는 1년)이\n'
 '지났을 때\n'
 '3. 계약을 체결한 날부터 3년이 지났을 때\n'
 '4. 회사가 이 계약을 청약할 때 반려동물의 건강상태를 판단할 수 있는 기초자료(건강\n'
 '진단서 사본 등)에 따라 승낙한 경우에 건강진단서 사본 등에 명기되어 있는 사항으'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000055',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
