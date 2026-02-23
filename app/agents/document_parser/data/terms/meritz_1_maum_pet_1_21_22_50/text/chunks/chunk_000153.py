from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 제1항 제1호에 따른 계약의 해지가 손해발생 후에 이루어진 경우에 회사는 그 손해를\n'
 '보상하지 않으며, 계약 전 알릴 의무 위반 사실뿐만 아니라 계약 전 알릴 의무사항이- 28 -중요한 사항에 해당되는 사유를「반대증거가 '
 '있는 경우 이의를 제기할 수 있습니다」\n'
 '라는 문구와 함께 계약자에게 서면 또는 전자문서 등으로 알려 드립니다. 또한 이 경우\n'
 '계약 해지로 인하여 회사가 환급하여야 할 보험료가 있을 때에는 보통약관 제33조(보\n'
 '험료의 환급)에 따른 보험료를 계약자에게 지급합니다. 회사가 전자문서로 안내하고자'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
