from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사가 제1조(보험금의 지급사유)에서 정한 6대호흡계특정질환진단비를 지급한 해\n'
 '- 경우에는 그 지급사유가 발생한 때부터 이 특별약관 계약은 소멸되며 이 특별약\n'
 '- 관의 해약환급금을 지급하지 않습니다.\n'
 '- \uf000 피보험자가 사망하였을 경우에는 이 특별약관 계약도 소멸되며 회사는 "보험료\n'
 '- 및 해약환급금 산출방법서"에서 정하는 바에 따라 피보험자의 사망 당시 이 특별\n'
 '- 약관의 계약자적립액 및 미경과보험료를 계약자에게 지급합니다.'),
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
