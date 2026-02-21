from langchain_core.documents import Document

chunk = Document(
    page_content=('- 및 미경과보험료를 계약자에게 지급합니다.\n'
 '# 제4조(준용규정)# 이특별약관에서 정하지 않은 사항은 보통약관 제1절 일반조항을 따릅니다. 다만,이 특별약관에서는 보통약관 제1절 '
 '일반조항 제9조(만기환급금의 지급), 제24조(계\n'
 '약의 소멸) 및 제36조(중도인출)는 제외합니다.90 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)# 2대호흡계특정질환진단비# '
 '2.제1조(보험금의 지급사유)\n'
 '회사는 피보험자가 이 특별약관의 보험기간 중에 2대호흡계특정질환으로 진단확정된'),
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
