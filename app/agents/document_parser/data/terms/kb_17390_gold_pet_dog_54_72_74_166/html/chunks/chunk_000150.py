from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다</p><br><p id='181' data-category='paragraph' style='font-size:14px'>만, "
 '다음 각 호의 어느 하나에 해당하는 계약은 청약을 철회할 수 없습니다.<br>1. 회사가 건강상태 진단을 지원하는 계약</p><br><p '
 "id='182' data-category='list' style='font-size:14px'>2. 보험기간이 90일 이내인 "
 '계약<br>3'),
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
