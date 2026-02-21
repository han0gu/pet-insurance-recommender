from langchain_core.documents import Document

chunk = Document(
    page_content=('- 비공휴일을 대체공휴일로 한다. 규정\n'
 '- ※ 향후 관련법령이 개정된 경우 개정된 내용을 적용합니다.\n'
 '- 55 -KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 55공보관통약특별약관제 2 관 보험금의 지급제3조(보험금의 지급사유)제2절 '
 '보통약관의 보장을 따릅니다.# 제4조(보험금 지급에 관한세부규정)# 제2절 보통약관의 보장을 따릅니다.제5조(보험금을 지급하지 않는 '
 '사유)\n'
 '\uf000 회사는 다음 중 어느 한가지로 보험금 지급사유가 발생한 때에는 보험금을 지급하- 지 않습니다.'),
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
