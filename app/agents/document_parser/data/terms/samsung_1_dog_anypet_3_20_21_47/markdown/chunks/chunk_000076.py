from langchain_core.documents import Document

chunk = Document(
    page_content=('약을 해지할 수 있습니다. 다만, 이 경우에도 회사는 이미 발생한 보험금 지급사유에 대해서는 보험금\n'
 '을 지급합니다.② 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취지를 계약자에게 통지하고 제30조(보험료의\n'
 '환급)에 따라 보험료를 계약자에게 지급합니다.# 제28조(회사의 파산선고와 해지)- ① 회사가 파산의 선고를 받은 때에는 계약자는 계약을 '
 '해지할 수 있습니다.\n'
 '- ② 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후 3개월이 지난 때에는 그 효력을 잃습니다.'),
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
