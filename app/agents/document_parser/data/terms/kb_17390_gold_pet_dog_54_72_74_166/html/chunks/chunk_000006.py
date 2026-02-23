from langchain_core.documents import Document

chunk = Document(
    page_content=('말합니다.</td></tr><tr><td>중요한 사항</td><td>계약전 알릴 의무와 관련하여 회사가 그 사실을 알았더라 면 계약의 '
 '청약을 거절하거나 보험가입금액 한도 제한, 일 부 보장 제외, 보험금 삭감, 보험료 할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 '
 '미칠 수 있는 사항을 말 합니다.</td></tr><tr><td>한국표준질병․ 사인분류</td><td>제9차 개정 '
 '한국표준질병․사인분류(KCD, 통계청 고시 제 2025-299호, 2026.1.1'),
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
