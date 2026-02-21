from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 바에 따라 보장을 합니다. 또한, 회사가 청약과 함께 제1회 보험료 등을 받은\n'
 '- 후 승낙한 경우에도 제1회 보험료 등을 받은 때부터 보장이 개시됩니다. 자동이\n'
 '- 체 또는 신용카드로 납입하는 경우에는 자동이체신청 또는 신용카드매출승인에\n'
 '- 필요한 정보를 제공한 때를 제1회 보험료 등을 받은 때로 하며, 계약자의 책임 있\n'
 '- 는 사유로 자동이체 또는 매출승인이 불가능한 경우에는 보험료가 납입되지 않은\n'
 '- 것으로 봅니다.\n'
 '- \uf000 회사가 청약과 함께 제1회 보험료 등을 받고 청약을 승낙하기 전에 보험금 지급사'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
