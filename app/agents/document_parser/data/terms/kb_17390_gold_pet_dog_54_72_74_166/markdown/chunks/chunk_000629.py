from langchain_core.documents import Document

chunk = Document(
    page_content=('관"에서 정한 방법에 따라 갱신된 계약의 무지개다리위로금보장개시일은 이 특별\n'
 '약관의 갱신일로 합니다.제2조(보험금 지급에 관한세부규정)보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 '
 '합의하지\n'
 '못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있\n'
 '습니다. 제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는'),
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
