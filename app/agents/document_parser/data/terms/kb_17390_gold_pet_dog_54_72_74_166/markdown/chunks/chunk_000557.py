from langchain_core.documents import Document

chunk = Document(
    page_content=('못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있\n'
 '습니다. 제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는# 의료비용은 회사가 전액 부담합니다.# 제3조(보험금을 '
 '지급하지 않는 사유)\uf000 회사는 아래의 사유로 인한 손해는 보상하지 않습니다.- 1. 계약자, 피보험자, 이들의 가족 또는 '
 '사용인의 고의 또는 중대한 과실\n'
 '- 2. 지진, 분화, 해일, 홍수 또는 이와 유사한 자연재해로 생긴 손해'),
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
