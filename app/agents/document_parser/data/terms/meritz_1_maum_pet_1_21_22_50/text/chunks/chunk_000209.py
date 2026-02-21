from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료 납입영수증에 장애인전용 보장성 보험료로 표시되지 않고 특별세액공제 대상에\n'
 '포함되지 않으며, 장애인전용보험으로 전환된 이후(2019년6월1일 ~ 2019년12월31일)\n'
 '납입된 보험료만 2019년 특별세액공제 대상이 됩니다.③ 제2항에도 불구하고,「전환대상계약이 장애인전용보험으로 전환된 당해 연도에 '
 '제4조\n'
 '(전환 취소)에 따라 전환을 취소하는 경우」에는 당해 연도에 납입한 모든 전환대상계\n'
 '약보험료가 보험료 납입영수증에 장애인전용 보장성보험료로 표시되지 않습니다. 다만,'),
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
