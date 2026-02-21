from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 동물병원에서 수의사에게 수술을 받은 경우 수술 당<br>일 발생한 수술비 및 치료비는 보상하여 드리지 않습니다.<br>② 회사가 '
 '보상하는 비용은 각 항목별 피보험자가 부담한 치료비에서 보험증권에 기재된<br>자기부담금을 차감한 후, 보험증권에 기재된 보상비율을 곱한 '
 '금액을 보험증권에서 정<br>한 1일당 지급 한도를 적용하여 보상합니다'),
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
