from langchain_core.documents import Document

chunk = Document(
    page_content=("- 이 공시하는 보험계약대출이율'을 연단위 복리로 계산한 금액을 더하여 지급합니다.\n"
 "- 회사가 해지권을 행사하는 경우 제4항의 '청구일'은 회사의 해지 의사표시(서면, 전자우편, 휴대전\n"
 '- 화 문자메시지 또는 이에 준하는 전자적 의사표시 포함)가 계약자 또는 그의 대리인에게 도달한 날\n'
 '- 로 봅니다.\n'
 '제7관 분쟁의 조정 등# 제31조(분쟁의 조정)- ① 계약에 관하여 분쟁이 있는 경우 분쟁 당사자 또는 기타 이해관계인과 회사는 '
 '금융감독원장에게'),
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
