from langchain_core.documents import Document

chunk = Document(
    page_content=('한 강제집행, 담보권실행, 국세 및 지방세 체납처분절차에 따라 계약이 해지된 경우에\n'
 '는, 회사는 해지 당시의 보험수익자가 계약자의 동의를 얻어 계약 해지로 회사가 채권\n'
 '자에게 지급한 금액을 회사에게 지급하고 제23조(계약내용의 변경 등) 제1항의 절차에\n'
 '따라 계약자 명의를 보험수익자로 변경하여 계약의 특별부활(효력회복)을 청약할 수 있\n'
 '음을 보험수익자에게 통지하여야 합니다.\n'
 '② 회사는 제1항에 따른 계약자 명의변경 신청 및 계약의 특별부활(효력회복) 청약을 승낙\n'
 '하며, 계약은 청약한 때부터 특별부활(효력회복) 됩니다.'),
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
