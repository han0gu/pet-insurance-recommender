from langchain_core.documents import Document

chunk = Document(
    page_content=('단확정 된 경우 이 특별약관의 보험가입금액을 골절진단비로 보험수익자에게 매 사\n'
 '고시마다 지급합니다.제2조(보험금 지급에 관한 세부규정)- \uf000 제1조(보험금의 지급사유)의 골절진단비는 같은 상해를 직접적인 '
 '원인으로 2가지\n'
 '- 이상의 골절 발생시에는 1회에 한하여 골절진단비를 지급합니다.\n'
 '- \uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의\n'
 '- 상\n'
 '- 하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따\n'
 '- 해'),
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
