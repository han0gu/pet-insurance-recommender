from langchain_core.documents import Document

chunk = Document(
    page_content=('질\n'
 '외모특정상해(머리,목)수술비로 보험수익자에게 매 사고시마다 지급합니다.\n'
 '병제2조(보험금 지급에 관한 세부규정)\uf000 제1조(보험금의 지급사유)의외모특정상해(머리,목)수술비는 같은 상해로 두 종상- 반\n'
 '- 류 이상의 외모특정상해(머리,목)수술을 받거나 같은 종류의 수술을 2회 이상 받\n'
 '- 려\n'
 '- 은 경우에는 하나의 외모특정상해(머리,목)수술비만 지급합니다.\n'
 '- 동\n'
 '- \uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의 물\n'
 '- 하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따'),
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
