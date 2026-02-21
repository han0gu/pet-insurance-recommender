from langchain_core.documents import Document

chunk = Document(
    page_content=('- 유로 회사에 이의를 제기할 수 없습니다.\n'
 '- 타인을 위한 계약에서 보험사고가 발생한 경우에 계약자가 그 타인에게 보험사고의 발생으로 생긴\n'
 '- 손해를 배상한 때에는 계약자는 그 타인의 권리를 해하지 않는 범위 안에서 회사에 보험금의 지급\n'
 '- 을 청구할 수 있습니다.\n'
 '제12조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 29 -당신에게 좋은보험 삼성화재# 반려동물 사망위로금 '
 '특별약관# 제1조(보상하는 손해)- ① 회사는 보험증권에 기재된 반려동물이 보험기간 중에 사망한 경우 보험증권에 기재된 보험가입금'),
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
