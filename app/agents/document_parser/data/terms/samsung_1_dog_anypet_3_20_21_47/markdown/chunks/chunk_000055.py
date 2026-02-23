from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 제12조(계약 전 알릴 의무)의 규정에 의하여 계약자 또는 피보험자가 회사에 알린 내용이 보험\n'
 '- 금 지급사유의 발생에 영향을 미쳤음을 회사가 증명하는 경우\n'
 '- 2. 제5조(보상하지 않는 손해), 제14조(사기에 의한 계약), 제18조(계약의 무효) 또는 제26조(계약\n'
 '- 의 해지)의 규정을 준용하여 회사가 보장을 하지 않을 수 있는 경우\n'
 '④ 계약자가 제1회 보험료 등을 자동이체 또는 신용카드로 납입하는 경우에는 자동이체신청 및 신용'),
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
