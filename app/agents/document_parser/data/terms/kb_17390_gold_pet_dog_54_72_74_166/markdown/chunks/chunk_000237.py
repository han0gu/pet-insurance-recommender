from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 전액 부담합니다.\n'
 '74 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- 제3조(특별약관의 소멸)\n'
 '- \uf000 회사는 제1조(보험금의 지급사유)에서 정한 반려동물양육자금Ⅰ(일반상해사망)\n'
 '- 을 지급한 때에는 그 지급사유가 발생한 때부터 이 특별약관은 소멸되며 이 특별\n'
 '- 약관의 해약환급금을 지급하지 않습니다.\n'
 '- \uf000 제1조(보험금의 지급사유)에서 정하지 않는 사유로 피보험자가 사망하였을 경우'),
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
