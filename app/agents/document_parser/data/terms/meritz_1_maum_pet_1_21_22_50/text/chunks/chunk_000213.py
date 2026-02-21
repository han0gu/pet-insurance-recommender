from langchain_core.documents import Document

chunk = Document(
    page_content=('범위) 제1항 제2호에서 정한 조건을 만족하지 않게 된 경우에는 이 조항이 적용되지\n'
 '않습니다.제4조(전환 취소)계약자는 전환대상계약에 대하여 장애인전용보험으로의 전환을 취소할 수 있으며, 이 경우\n'
 '전환취소 신청서를 회사에 제출하여야 합니다.제5조(준용규정)① 이 특별약관에서 정하지 않은 사항에 대하여는 전환대상계약의 약관, 소득세법 '
 '등 관련\n'
 '법규에서 정하는 바에 따릅니다.'),
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
