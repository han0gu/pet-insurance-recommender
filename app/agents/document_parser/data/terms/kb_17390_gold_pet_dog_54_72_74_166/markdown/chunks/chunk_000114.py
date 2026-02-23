from langchain_core.documents import Document

chunk = Document(
    page_content=('도장을 찍는 날인과 전자서명법 제2조 제2호에 따른 전자서명을 포함합니다)\n'
 '\uf000 제3항에도 불구하고 전화를 이용하여 계약을 체결하는 경우 다음의 각 호의 어느\n'
 '공\n'
 '하나를 충족하는 때에는 자필서명을 생략할 수 있으며, 제2항의 규정에 따른 음성\n'
 '통녹음 내용을 문서화한 확인서를 계약자에게 드림으로써 계약자 보관용 청약서를전달한 것으로 봅니다.# 1. 계약자, 피보험자 및 '
 '보험수익자가 동일한 계약의 경우2. 계약자, 피보험자가 동일하고 보험수익자가 계약자의 법정상속인인 계약일 경우'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
