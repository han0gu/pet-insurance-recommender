from langchain_core.documents import Document

chunk = Document(
    page_content=('. 초회보험료자동납입 추가특별약관</td></tr><tr><td>제1조(보험료의 납입) \uf000 계약자가 제1회 보험료의 납입방법을 '
 '계약자의 거래은행 지정 계좌를 통한 자동납 입으로 가입하고자 하는 경우에, 회사는 청약서를 접수하고 자동이체신청에 필요 한 정보를 제공한 '
 '때를 청약일 및 제1회 보험료 납입일로 하여 보통약관 제1절 일 반조항 제18조(보험계약의 성립)의 규정을 적용합니다'),
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
