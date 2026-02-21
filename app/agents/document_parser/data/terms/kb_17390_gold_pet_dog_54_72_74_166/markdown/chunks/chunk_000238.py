from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1조(보험금의 지급사유)에서 정하지 않는 사유로 피보험자가 사망하였을 경우\n'
 '- 에는 이 특별약관도 소멸되며 회사는 "보험료 및 해약환급금 산출방법서"에서 정\n'
 '- 하는 바에 따라 피보험자의 사망 당시 이 특별약관의 계약자적립액 및 미경과보험\n'
 '# 료를 계약자에게 지급합니다.제4조(준용규정)\n'
 '이 특별약관에서 정하지 않은 사항은 보통약관 제1절 일반조항을 따릅니다. 다만,'),
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
