from langchain_core.documents import Document

chunk = Document(
    page_content=('- 치료를 직접적인 목적으로 동물병원에 통원 또는 입원하여 수의사에게 치료를 받은 때\n'
 '- 에는 피보험자가 부담한 반려동물의 치료비를 이 약관에 따라 피보험자에게 치료비보\n'
 '- 험금으로 보상하여 드립니다. 단, 동물병원에서 수의사에게 수술을 받은 경우 수술 당\n'
 '- 일 발생한 수술비 및 치료비는 보상하여 드리지 않습니다.\n'
 '- ② 회사가 보상하는 비용은 각 항목별 피보험자가 부담한 치료비에서 보험증권에 기재된\n'
 '- 자기부담금을 차감한 후, 보험증권에 기재된 보상비율을 곱한 금액을 보험증권에서 정'),
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
