from langchain_core.documents import Document

chunk = Document(
    page_content=('- ④ 피보험자는 통지를 받은 날부터 15일 이내에 제1항의 절차를 이행할 수 있습니다.\n'
 '- 15 -당신에게 좋은보험 삼성화재# 제6관 계약의 해지 및 보험료의 환급 등# 제26조(계약의 해지)- ① 계약자는 손해가 발생하기 '
 '전에는 언제든지 계약을 해지할 수 있습니다. 다만, 타인을 위한 계약의\n'
 '- 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여 계약을 해지할 수\n'
 '- 있습니다.\n'
 '- ② 회사는 계약자 또는 피보험자의 고의로 손해가 발생한 경우 이 계약을 해지할 수 있습니다.'),
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
