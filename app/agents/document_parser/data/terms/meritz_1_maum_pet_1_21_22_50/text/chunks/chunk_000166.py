from langchain_core.documents import Document

chunk = Document(
    page_content=('납보험료는 변경적용하지 않습니다. 다만, 보통약관 제16조(계약 후 알릴 의무)에 따라\n'
 '보험료가 변경된 경우에는 예외로 합니다.제3조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 34 -보험료 자동납입 '
 '특별약관제1조(보험료의 납입)계약자는 보험료 분납 특별약관에 의하여 제2회 이후의 보험료부터 이 특별약관에 따라\n'
 '계약자의 거래은행 지정계좌를 이용하여 보험료를 자동 납입합니다.제2조(자동납입 신청)계약자는 보험계약과 동시에 계약자의 거래은행 '
 '지정계좌를 이용하여 보험료를 자동 납입'),
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
