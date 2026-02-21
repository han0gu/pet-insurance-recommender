from langchain_core.documents import Document

chunk = Document(
    page_content=('실대로 고지하지 않게 하였거나 부실한 고지를 권유했을 때. 다만, 보험설계사 등의\n'
 '행위가 없었다 하더라도 계약자 또는 피보험자가 사실대로 고지하지 않거나 부실한\n'
 '고지를 했다고 인정되는 경우에는 계약을 해지할 수 있습니다.③ 제1항에 따라 계약의 해지가 보험금 지급사유 발생 전에 이루어진 경우, '
 '이로 인하여\n'
 '회사가 환급하여야 할 보험료가 있을 때에는 제33조(보험료의 환급)에 따른 보험료를\n'
 '계약자에게 지급합니다.\n'
 '④ 제1항 제1호에 의한 계약의 해지가 보험금 지급사유 발생 후에 이루어진 경우에 회사'),
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
