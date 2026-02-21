from langchain_core.documents import Document

chunk = Document(
    page_content=('- 알릴 의무를 계약자 또는 피보험자의 고의 또는 중대한 과실로 이행하지 않았을\n'
 '- 때\n'
 '② 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 이 특별\n'
 '약관을 해지할 수 없습니다.- 1. 회사가 최초계약 체결당시에 그 사실을 알았거나 과실로 인하여 알지 못하였을 때\n'
 '- 2. 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또는 최초계약의 제1회 보험료\n'
 '- 를 받은 때부터 보험금 지급사유가 발생하지 않고 2년(진단계약의 경우 질병에 대\n'
 '- 하여는 1년)이 지났을 때'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
