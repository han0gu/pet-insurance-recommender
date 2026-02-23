from langchain_core.documents import Document

chunk = Document(
    page_content=('법KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 65- 65 -제 5 관 보험료의 납입- 제 25조(제1회 보험료 및 회사의 '
 '보장개시)\n'
 '- \uf000 회사는 계약의 청약을 승낙하고 제1회 보험료를 받은 때부터 이 약관이 정한 바에\n'
 '- 따라 보장을 합니다. 또한, 회사가 청약과 함께 제1회 보험료를 받은 후 승낙한\n'
 '- 경우에도 제1회 보험료를 받은 때부터 보장이 개시됩니다. 자동이체 또는 신용카\n'
 '- 드로 납입하는 경우에는 자동이체신청 또는 신용카드매출승인에 필요한 정보를 제'),
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
