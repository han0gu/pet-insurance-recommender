from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 피보험자가 이 특별약관의 보험기간 중에 상해 또는 진단확정된 질병으로 인\n'
 '하여 "깁스(Cast)치료"를 받은 경우 이 특별약관 보험가입금액을 깁스치료비로 보험수익자에게 매 사고시마다 지급합니다.- 제2조(보험금 '
 '지급에 관한 세부규정)\n'
 '- \uf000 제1조(보험금의 지급사유)의 깁스치료비는 같은 상해 또는 질병으로 인하여 깁스\n'
 '- 치료를 2회 이상 받은 경우, 또는 동시에 서로 다른 신체부위에 깁스치료를 받은\n'
 '- 경우에는 1회에 한하여 깁스치료비를 지급합니다.'),
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
