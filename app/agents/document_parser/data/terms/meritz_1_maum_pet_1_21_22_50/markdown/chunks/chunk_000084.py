from langchain_core.documents import Document

chunk = Document(
    page_content=('- 라 보장을 합니다. 또한, 회사가 청약과 함께 제1회 보험료를 받은 후 승낙한 경우에도\n'
 '- 제1회 보험료를 받은 때부터 보장이 개시됩니다. 자동이체 또는 신용카드로 납입하는\n'
 '- 경우에는 자동이체신청 또는 신용카드매출승인에 필요한 정보를 제공한 때를 제1회 보\n'
 '- 험료를 받은 때로 하며, 계약자의 책임 있는 사유로 자동이체 또는 매출승인이 불가능\n'
 '- 한 경우에는 보험료가 납입되지 않은 것으로 봅니다.\n'
 '- ② 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙하기 전에 보험금 지급사유가 발'),
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
