from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기합니다.\n'
 '- ④ 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대한 것인 경우에\n'
 '- 는 그 권리를 취득하지 못합니다. 다만, 손해가 그 가족의 고의로 인하여 발생한 경우에는 그 권리\n'
 '- 를 취득합니다.\n'
 '- 28 -당신에게 좋은보험 삼성화재# 제11 조(타인을 위한 계약)- 계약자는 타인을 위한 계약을 체결하는 경우에 그 타인의 위임이 '
 '없는 때에는 반드시 이를 회사에\n'
 '- 알려야 하며, 이를 알리지 않았을 때에는 그 타인은 이 계약이 체결된 사실을 알지 못하였다는 사'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
