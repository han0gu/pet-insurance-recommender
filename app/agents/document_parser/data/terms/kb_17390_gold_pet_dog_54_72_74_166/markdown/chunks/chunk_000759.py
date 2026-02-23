from langchain_core.documents import Document

chunk = Document(
    page_content=('단 결과 피보험자의 남은 생존기간이 6개월 이내라고 판단한 경우에 회사의 신청\n'
 '서에 정한 바에 따라 사망보험금의 50%를 선지급 사망보험금(이하 "보험금"이라- 132 -- 합니다)으로 피보험자에게 지급합니다.\n'
 '- \uf000 이 특별약관의 보험금을 지급하였을 때에는 지급한 보험금액에 해당하는 계약의 보\n'
 '- 험가입금액이 지급일에 감액된 것으로 봅니다. 다만, 그 감액부분에 해당하는 해\n'
 '- 약환급금이 있어도 이를 지급하지 않습니다. 이 경우 이 특별약관의 보험금 지급'),
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
