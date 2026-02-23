from langchain_core.documents import Document

chunk = Document(
    page_content=('- 규정에서 정한 지급예정일을 통지한 경우를 포함합니다)에는 그 다음날부터 지급\n'
 '- 일까지의 기간에 대하여 "보험금을 지급할 때의 적립이율 계산"(【별표2】참조)\n'
 '- 에서 정한 이율로 계산한 금액을 보험금에 더하여 지급합니다. 그러나 계약자,\n'
 '- 피보험자 또는 보험수익자의 책임 있는 사유로 지급이 지연된 때에는 그 해당기\n'
 '- 간에 대한 이자는 더하여 지급하지 않습니다. 다만, 회사는 계약자 등이 분쟁조\n'
 '- 정을 신청했다는 사유만으로 이자지급을 거절하지 않습니다.'),
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
