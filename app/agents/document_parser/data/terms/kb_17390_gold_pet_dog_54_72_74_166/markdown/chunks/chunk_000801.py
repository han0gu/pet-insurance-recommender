from langchain_core.documents import Document

chunk = Document(
    page_content=('| ∙ 소득세법 시행령 제107조(장애인의 범위)에서 규정한 장애인 1. "장애인복지법"에 따른 장애인 및 "장애아동 복지지원법"에 따른 '
 '장애 아동 중 기획재정부령으로 정하는 사람 2. "국가유공자 등 예우 및 지원에 관한 법률"에 의한 상이자 및 이와 유 사한 사람으로서 '
 '근로능력이 없는 사람 3'),
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
