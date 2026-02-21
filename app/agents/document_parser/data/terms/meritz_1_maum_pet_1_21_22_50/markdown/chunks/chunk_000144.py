from langchain_core.documents import Document

chunk = Document(
    page_content=('- 요구가 있으면 피보험자 및 계약자는 필요한 서류·증거의 제출, 증언 또는 증인 출석에\n'
 '- 협조하여야 합니다.\n'
 '- ③ 피보험자가 피해자로부터 손해배상의 청구를 받았을 경우에 회사가 필요하다고 인정할\n'
 '- 때에는 피보험자를 대신하여 회사의 비용으로 이를 해결할 수 있습니다. 이 경우 회사\n'
 '- 의 요구가 있으면 계약자 및 피보험자는 이에 협력하여야 합니다.\n'
 '- ④ 계약자 및 피보험자가 정당한 이유없이 제2항 및 제3항의 요구에 협조하지 않은 때에\n'
 '- 는 회사는 그로 인하여 늘어난 손해는 보상하지 않습니다.'),
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
