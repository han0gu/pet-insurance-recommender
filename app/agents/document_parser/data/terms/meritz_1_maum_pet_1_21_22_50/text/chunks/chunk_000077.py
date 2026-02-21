from langchain_core.documents import Document

chunk = Document(
    page_content=('한 경우 회사는 변경 전 보험수익자에게 보험금을 지급할 수 있습니다. 회사가 변경\n'
 '전 보험수익자에게 보험금을 지급한 경우 변경된 보험수익자에게는 별도로 보험금을\n'
 '지급하지 않습니다.③ 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이상 지난 유효한 계약으로서 그\n'
 '보험종목의 변경을 요청할 때에는 회사의 사업방법서에서 정하는 방법에 따라 이를 변\n'
 '경하여 드립니다.\n'
 '④ 회사는 계약자가 제1항 제5호에 따라 보험가입금액을 감액하고자 할 때에는 그 감액된'),
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
