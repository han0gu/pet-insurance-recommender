from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 피보험자가 이 특별약관의 보험기간 중에 제3조(호흡기관련질병의 정의 및\n'
 '진단확정)에서 정한 "호흡기관련질병"으로 진단확정되고 그 치료를 직접적인 목적으\n'
 '로 수술을 받은 때에는 수술 1회당 이 특별약관의 보험가입금액을 호흡기관련질병# 수술비로 보험수익자에게 지급합니다.제2조(보험금 지급에 '
 '관한 세부규정)\n'
 '보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지\n'
 '못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있'),
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
