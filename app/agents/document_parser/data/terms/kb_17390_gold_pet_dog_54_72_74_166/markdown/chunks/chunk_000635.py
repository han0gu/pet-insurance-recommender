from langchain_core.documents import Document

chunk = Document(
    page_content=('- 필요하여 제출하는 서류 려동\n'
 '- 물\n'
 '병제5조(특별약관의 소멸)\uf000 회사는 제1조(보험금의- 지급한 경우에는 그 지급사유가 발생한 때부터 이 특별약관 계약은 소멸되며 '
 '이\n'
 '- 도\n'
 '- 특별약관의 해약환급금을 지급하지 않습니다.\n'
 '- 성\n'
 '- \uf000 보험증권에 기재된 반려동물이 보험기간 중에 이 특별약관에서 보장하지 않는 사\n'
 '- 특\n'
 '- 유로 사망하였을 경우 회사는 "보험료 및 해약환급금 산출방법서"에서 정하는 바 약'),
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
