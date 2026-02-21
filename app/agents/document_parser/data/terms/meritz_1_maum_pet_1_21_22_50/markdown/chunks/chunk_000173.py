from langchain_core.documents import Document

chunk = Document(
    page_content=('사에 알려야 합니다.제5조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 35 -# 초회보험료자동납입 '
 '추가특별약관제1조(보험료의 납입)- ① 보험계약자는 제1회 보험료의 납입방법을 보험계약자의 거래은행 지정계좌를 통한 자\n'
 '- 동납입으로 가입하고자 하는 경우에, 회사는 청약서를 접수하고 자동이체신청에 필요한\n'
 '- 정보를 제공한 때(다만, 보험계약자의 귀책사유로 보험료의 납입이 불가능한 경우에는\n'
 '- 거래은행의 지정계좌로부터 제1회 보험료가 이체된 날을 기준으로 합니다)를 청약일 및'),
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
