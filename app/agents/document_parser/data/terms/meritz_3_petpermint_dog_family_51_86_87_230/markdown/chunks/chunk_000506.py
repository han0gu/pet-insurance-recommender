from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계약을 다른 보험자와 맺으려고 하든지 또는 이와 같\n'
 '- 은 계약이 있음을 알았을 때\n'
 '- ③ 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때\n'
 '\uf000 회사는 제1항에 따라 위험이 감소된 경우에는 그 차액보\n'
 '험료를 돌려 드리며, 위험이 증가된 경우에는 통지를 받은\n'
 '날부터 1개월 내에 보험료의 증액을 청구하거나 계약을 해\n'
 '지할 수 있습니다.\n'
 '\uf000 계약자 또는 피보험자는 주소 또는 연락처가 변경된 경\n'
 '우에는 지체없이 이를 회사에 알려야 합니다. 다만, 계약자\n'
 '가 알리지 않은 경우 회사가 알고 있는 최종의 주소 또는'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
