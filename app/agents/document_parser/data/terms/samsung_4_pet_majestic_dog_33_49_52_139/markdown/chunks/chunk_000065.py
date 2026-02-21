from langchain_core.documents import Document

chunk = Document(
    page_content=('- 않았을 때\n'
 '② 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 계약을- 39 -# 해지할 수 없습니다.- 1. 회사가 '
 '최초계약 체결당시에 그 사실을 알았거나 과실로 인하여 알지 못하였을 때\n'
 '- 2. 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또는 제1회 보험료를 받은 때부\n'
 '- 터 보험금 지급사유가 발생하지 않고 2년(진단계약의 경우 질병에 대하여는 1년)\n'
 '- 이 지났을 때\n'
 '- 3. 최초계약을 체결한 날(갱신형 계약의 경우 최초 계약해당일을 말합니다)부터 3년이\n'
 '- 지났을 때'),
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
