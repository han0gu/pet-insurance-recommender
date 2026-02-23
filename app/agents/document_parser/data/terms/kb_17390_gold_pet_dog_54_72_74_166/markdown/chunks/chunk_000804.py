from langchain_core.documents import Document

chunk = Document(
    page_content=('| <table><thead></thead><tbody><tr><td>예</td></tr><tr><td>시 이 특별약관을 적용할 수 없는 '
 '사례 ∙ 전환대상계약의 피보험자 1인은 비장애인이고 보험수익자 2인 중 한명은 특 비장애인, 한명은 장애인인 경우 별 ⇒ 모든 '
 '보험수익자가 장애인이 아니므로 이 특별약관을 적용할 수 없습니다. 약 ∙ 전환대상계약의 보험수익자 1인은 비장애인이고 피보험자 2인 중 '
 '한명은 관 비장애인, 한명은 장애인인 경우 ⇒ 모든 피보험자가 장애인이 아니므로 이 특별약관을 적용할 수 없습니다'),
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
