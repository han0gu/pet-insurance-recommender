from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자는 계약이 성립한 날부터 3개월 이내<br>에 계약을 취소할 수 있습니다.<br>④ 제3항에도 불구하고 전화를 이용하여 계약을 '
 '체결하는 경우 다음의 각 호의 어느 하나<br>를 충족하는 때에는 자필서명을 생략할 수 있으며, 제2항의 규정에 따른 음성녹음 '
 "내용<br>을 문서화한 확인서를 계약자에게 드림으로써 계약자 보관용 청약서를 전달한 것으로<br>봅니다.</p><br><p id='24' "
 "data-category='list' style='font-size:14px'>1"),
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
