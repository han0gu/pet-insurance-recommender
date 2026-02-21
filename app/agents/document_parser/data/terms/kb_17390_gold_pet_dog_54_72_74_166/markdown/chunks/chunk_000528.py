from langchain_core.documents import Document

chunk = Document(
    page_content=('약자에게 지급합니다. 다만, 타인을 위한 계약의 경우에는 계약자는 그 타인의 동의 특\n'
 '를 얻거나 보험증권을 소지한 경우에 한하여 특별약관을 해지할 수 있습니다. 약전에는 언제든지 계약을 해지할 수 있으며, 이 경우질및병KB '
 '금쪽같은 펫보험(강아지)(무배당)(26.01) 107제- 107 -도# 제20조(중대사유로 인한 해지)- \uf000 회사는 아래와 같은 '
 '사실이 있을 경우에는 안 날부터 1개월 이내에 계약을 해지할\n'
 '- 수 있습니다.\n'
 '- 1. 계약자, 피보험자 또는 보험수익자가 보험금을 지급받을 목적으로 고의로 보험'),
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
