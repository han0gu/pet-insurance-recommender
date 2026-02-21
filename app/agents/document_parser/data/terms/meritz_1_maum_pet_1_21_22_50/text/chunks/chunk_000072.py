from langchain_core.documents import Document

chunk = Document(
    page_content=('자가 청약서에 자필서명(날인(도장을 찍음) 및 ⌜전자서명법⌟ 제2조 제2호에 따른 전\n'
 '자서명을 포함합니다)을 하지 않은 때에는 계약자는 계약이 성립한 날부터 3개월 이내\n'
 '에 계약을 취소할 수 있습니다.\n'
 '④ 제3항에도 불구하고 전화를 이용하여 계약을 체결하는 경우 다음의 각 호의 어느 하나\n'
 '를 충족하는 때에는 자필서명을 생략할 수 있으며, 제2항의 규정에 따른 음성녹음 내용\n'
 '을 문서화한 확인서를 계약자에게 드림으로써 계약자 보관용 청약서를 전달한 것으로\n'
 '봅니다.1. 계약자, 피보험자 및 보험수익자가 동일한 계약의 경우'),
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
