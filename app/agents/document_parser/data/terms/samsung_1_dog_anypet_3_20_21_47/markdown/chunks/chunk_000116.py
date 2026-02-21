from langchain_core.documents import Document

chunk = Document(
    page_content=('우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 보상책임액의 합계액이 손해액을\n'
 '초과할 때에는 회사는 아래에 따라 손해를 보상합니다. 이 계약과 다른 계약이 모두 의무보험인 경\n'
 '우에도 같습니다.이 계약의 보상책임액\n'
 '손해액 ×\n'
 '다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액- ② 이 계약이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 '
 '의무보험에서 보상되는 금액(피\n'
 '- 보험자가 가입을 하지 않은 경우에는 보상될 것으로 추정되는 금액)을 차감한 금액을 손해액으로'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
