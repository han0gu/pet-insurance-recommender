from langchain_core.documents import Document

chunk = Document(
    page_content=('| 용 어 | 정 의 |\n'
 '| 계약자 | 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사 람을 말합니다. |\n'
 '| 보험수익자 | 보험금 지급사유가 발생하는 때에 회사에 보험금을 청구하 여 받을 수 있는 사람을 말합니다. 그리고 만기환급금 지급 '
 '시기에 만기환급금을 청구하여 받을 수 있는 사람을 말합 니다. |\n'
 '| 보험증권 | 계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자 에게 드리는 증서를 말합니다. |\n'
 '| 진단계약 | 계약을 체결하기 위하여 피보험자가 건강진단을 받아야 하 는 계약을 말합니다. |'),
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
