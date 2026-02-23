from langchain_core.documents import Document

chunk = Document(
    page_content=('연도에 제4조<br>(전환 취소)에 따라 전환을 취소하는 경우」에는 당해 연도에 납입한 모든 전환대상계<br>약보험료가 보험료 '
 '납입영수증에 장애인전용 보장성보험료로 표시되지 않습니다'),
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
