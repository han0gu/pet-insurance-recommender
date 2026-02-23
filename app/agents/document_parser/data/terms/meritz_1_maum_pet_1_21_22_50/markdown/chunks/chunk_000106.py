from langchain_core.documents import Document

chunk = Document(
    page_content=('- 18 -우에 회사는 제33조(보험료의 환급)에 따른 보험료를 계약자에게 지급합니다.# 제33조(보험료의 환급)① 이 계약이 무효, '
 '효력상실, 해지 또는 소멸된 때에는 다음과 같이 보험료를 돌려드립니\n'
 '다.- 1. 계약자, 피보험자 또는 보험수익자의 책임없는 사유에 의하는 경우 : 무효의 경우에\n'
 '- 는 회사에 납입한 보험료의 전액, 효력상실, 해지 또는 소멸의 경우에는 경과하지\n'
 '- 아니한 기간에 대하여 일단위로 계산한 보험료\n'
 '- 2. 계약자, 피보험자 또는 보험수익자의 책임있는 사유에 의하는 경우 : 이미 경과한'),
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
