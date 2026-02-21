from langchain_core.documents import Document

chunk = Document(
    page_content=('- ④ 제1항에 따라 제출한 장애인증명서의 장애기간이 변경되는 경우 계약자는 이를 회사에\n'
 '- 알리고 변경된 장애기간이 기재된 장애인증명서를 제출하여야 합니다.\n'
 '# 제3조(장애인전용보험으로의 전환)- ① 회사는 이 특별약관이 부가된 전환계약을「소득세법 제59조의4(특별세액공제) 제1항\n'
 '- 제1호」에 해당하는 장애인전용보험으로 전환하여 드립니다.\n'
 '- ② 제1항에 따라 전환대상계약이 장애인전용보험으로 전환된 후부터 납입된 전환대상계약\n'
 '- 보험료는 보험료 납입영수증에 장애인전용 보장성보험료로 표시됩니다.'),
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
