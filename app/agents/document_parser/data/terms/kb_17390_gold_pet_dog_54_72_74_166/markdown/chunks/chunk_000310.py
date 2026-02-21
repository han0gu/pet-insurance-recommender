from langchain_core.documents import Document

chunk = Document(
    page_content=('- 퇴원하기 전까지의 계속 중인 입원에 대하여는 제1조(보험금의 지급사유) 제2항\n'
 '- 에 따라 상해입원일당을 계속 지급합니다.\n'
 '- \uf000 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회\n'
 '- 사는 상해입원일당의 전부 또는 일부를 지급하지 않습니다.\n'
 '- \uf000 피보험자가 병원 또는 의원을 이전하여 입원한 경우에도 동일한 상해의 치료를\n'
 '- 목적으로 2회 이상 입원한 경우에는 계속하여 입원한 것으로 보아 각 입원일수를\n'
 '- 더합니다.\n'
 '- \uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의'),
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
