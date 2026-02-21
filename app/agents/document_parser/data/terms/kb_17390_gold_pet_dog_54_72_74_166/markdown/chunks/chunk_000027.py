from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제2항에 의하여 장해지급률의 판정 및 지급할 보험금의 결정과 관련하여 확정된 장\n'
 '- 해지급률에 따른 보험금을 초과한 부분에 대한 분쟁으로 보험금 지급이 늦어지는\n'
 '- 경우에는 보험수익자의 청구에 따라 이미 확정된 보험금을 먼저 가지급합니다.\n'
 '- \uf000 제2항에 의하여 추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에 따\n'
 '| 라 회사가 추정하는 |  |  | 보험금의 50% 상당액을 가지급보험금으로 지급합니다. |\n'
 '| --- | --- | --- | --- |'),
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
