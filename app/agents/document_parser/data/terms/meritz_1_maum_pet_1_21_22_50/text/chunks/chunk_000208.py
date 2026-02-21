from langchain_core.documents import Document

chunk = Document(
    page_content=('제1호」에 해당하는 장애인전용보험으로 전환하여 드립니다.\n'
 '② 제1항에 따라 전환대상계약이 장애인전용보험으로 전환된 후부터 납입된 전환대상계약\n'
 '보험료는 보험료 납입영수증에 장애인전용 보장성보험료로 표시됩니다.【예시】2019년 1월 15일에 전환대상계약에 가입한 계약자가 2019년 '
 '6월 1일에 이 특별약관을\n'
 '청약하고 회사가 승낙하여 전환대상계약이 장애인전용보험으로 전환된 경우, 이 특별약\n'
 '관을 청약하기 전(2019년 1월 15일 ~ 2019년 5월 31일)에 납입된 보험료는 당해 연도'),
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
