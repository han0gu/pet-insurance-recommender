from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이하 같습니다)에 부가하여 이루어집니다. 단, 제2호에 해당하는</p><br><p id='114' "
 "data-category='list' style='font-size:14px'>경우 계약자의 동의가 필요합니다.<br>1. 보험계약을 "
 '체결할 때 해당 반려동물의 건강상태가 보험회사가 정한 기준에 적<br>합하지 않은 경우<br>2'),
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
