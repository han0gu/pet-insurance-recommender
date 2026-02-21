from langchain_core.documents import Document

chunk = Document(
    page_content=('- 또는 이와 같은 계약이 있음을 알았을 때\n'
 '- 3. 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때\n'
 '- 4. 양도할 때\n'
 '- ② 회사는 제1항에 따라 위험이 감소된 경우에는 그 차액보험료를 돌려드리며, 위험이 증가된 경우에\n'
 '- 는 통지를 받은 날부터 1개월 이내에 보험료의 증액을 청구하거나 계약을 해지할 수 있습니다.\n'
 '- ③ 계약자 또는 피보험자는 주소 또는 연락처가 변경된 경우에는 지체없이 이를 회사에 알려야 합니\n'
 '- 다. 다만, 계약자 또는 피보험자가 알리지 않은 경우 회사가 알고 있는 최종의 주소 또는 연락처로'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
