from langchain_core.documents import Document

chunk = Document(
    page_content=('- 위)에서 정하는 동물위탁관리업자로써, 반려동물 소유자의 위탁을 받아 반려동물\n'
 '- 을 영업장 내에서 일시적으로 사육, 훈련 또는 보호하는 영업을 행하는 시설을\n'
 '- 말합니다.\n'
 '- \uf000 제1항의 반려동물 위탁비용의 지급일수는 1회 입원당 180일을 최고한도로 하며\n'
 '- 피보험자의 입원기간을 초과할 수 없습니다. 질\n'
 '- \uf000 제1항의 반려동물 위탁비용은 위탁1일당 이 특별약관의 보험가입금액을 한도로 병\n'
 '- 합니다.\n'
 '# 제2조(보험금 지급에 관한 세부규정)\uf000 제1조(보험금의 지급사유) 제1항의 반려동물 위탁비용은 같은 질병의 치료를 목 상'),
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
