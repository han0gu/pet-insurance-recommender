from langchain_core.documents import Document

chunk = Document(
    page_content=('료의 일부를 부담한 경우에 한하여 탈퇴일로부터 1개월 이내에 계약자 또는 피보험자\n'
 '는 회사의 승낙을 얻어 개별계약으로 전환할 수 있으며, 이 경우 피보험자는 개별계약의\n'
 '계약자가 됩니다.\n'
 '② 제1항에 따라 개별계약으로 전환시에는 전환후 피보험자의 보험기간은 이 계약의 남은\n'
 '기간으로 하고, 이로 인하여 발생하는 추가 또는 환급되는 보험료는 보험료 및 해약환'),
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
