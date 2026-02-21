from langchain_core.documents import Document

chunk = Document(
    page_content=('는 것을 말합니다.# 제28조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))① 제27조(보험료의 납입이 연체되는 경우 '
 '납입최고(독촉)와 계약의 해지)에 따라 계약이\n'
 '해지되었으나 계약자가 제33조(보험료의 환급)에 따라 보험료를 돌려받지 않은 경우 계\n'
 '약자는 해지된 날부터 3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을\n'
 '청약할 수 있습니다. 이 경우 회사가 그 청약을 승낙한 때에는 계약자는 부활(효력회'),
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
