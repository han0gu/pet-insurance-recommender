from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기간에 대하여 "보험금을 지급할 때의 적립이율 계산(【별표2】참조)"에서 정한\n'
 '- 이율로 계산한 금액을 보험금에 더하여 지급합니다. 그러나 피보험자의 책임있는\n'
 '- 사유로 지체된 경우에는 그 해당기간에 대한 이자를 더하여 지급하지 않습니다.\n'
 '- 다만, 회사는 피보험자가 분쟁조정을 신청했다는 사유만으로 이자지급을 거절하\n'
 '- 지 않습니다.\n'
 '제9조(보험금 등의 지급한도)# 회사는 1회의 보험사고에대하여 다음과 같이 보상합니다. 이 경우 보상한도액과 자기부담금은 각각 보험증권에 '
 '기재된 금액을 말합니다.'),
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
