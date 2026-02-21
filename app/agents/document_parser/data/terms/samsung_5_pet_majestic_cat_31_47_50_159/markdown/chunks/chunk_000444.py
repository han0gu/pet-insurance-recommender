from langchain_core.documents import Document

chunk = Document(
    page_content=('- 있는 사람을 말합니다.\n'
 '- 3. 보험증권: 계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자에게 드리는 증\n'
 '- 서를 말합니다.\n'
 '- 4. 진단계약: 계약을 체결하기 위하여 반려묘가 건강진단을 받아야 하는 계약을 말합\n'
 '- 니다.\n'
 '- 5. 피보험자: 반려묘의 소유와 관련하여 보험사고로 손해를 입은 사람을 말합니다.\n'
 '- 6. 반려묘 : 보험증권에 기재된 반려묘를 말하며, 이 특별약관에서 가입 가능한 반려\n'
 '- 묘는 대한민국 내에서 피보험자와 거주를 함께하고 있는 고양이(猫)를 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
