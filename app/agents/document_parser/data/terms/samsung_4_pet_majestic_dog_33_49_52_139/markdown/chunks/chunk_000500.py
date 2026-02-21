from langchain_core.documents import Document

chunk = Document(
    page_content=('약관을 해지할 수 없습니다.- 1. 회사가 최초계약 체결당시에 그 사실을 알았거나 과실로 인하여 알지 못하였을 때\n'
 '- 2. 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또는 최초계약의 제1회 보험료\n'
 '- 를 받은 때부터 보험금 지급사유가 발생하지 않고 2년(진단계약의 경우 질병에 대\n'
 '- 하여는 1년)이 지났을 때\n'
 '- 3. 최초계약을 체결한 날(재가입형 계약의 경우 최초 계약해당일을 말합니다)부터 3년\n'
 '- 이 지났을 때\n'
 '- 4. 회사가 이 계약을 청약할 때 반려견의 건강상태를 판단할 수 있는 기초자료(건강진'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
