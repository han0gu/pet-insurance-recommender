from langchain_core.documents import Document

chunk = Document(
    page_content=('전환대상계약에 대하여 장애인전용보험으로의 전환을 취소할 수 있으며, 이 경우<br>전환취소 신청서를 회사에 제출하여야 '
 "합니다.</p><h1 id='78' style='font-size:14px'>제5조(준용규정)</h1><br><p id='79' "
 "data-category='list' style='font-size:14px'>① 이 특별약관에서 정하지 않은 사항에 대하여는 "
 '전환대상계약의 약관, 소득세법 등 관련<br>법규에서 정하는 바에 따릅니다.<br>② 소득세법 등 관련법규가 제·개정 또는 폐지되는 경우 '
 '변경된 법령을'),
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
