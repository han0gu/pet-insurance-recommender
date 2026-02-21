from langchain_core.documents import Document

chunk = Document(
    page_content=('는 피보험자가 상법 제657조 제1항에 의해 보험사고의 발생을 회사에 알린 경우 약\n'
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 121- 121 -에는 제4조(보상하는 손해의 범위) 제1호 및 제2호 "다"목 또는 '
 '"라"목의 비용에# 대하여보상한도액을 한도로 보상하여 드립니다.제7조(보험금의 청구)- 피보험자가 보험금을 청구할 때에는 다음의 서류를 '
 '회사에 제출하여야 합니다.\n'
 '- 1. 보험금 청구서(회사양식)\n'
 '- 2. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본'),
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
