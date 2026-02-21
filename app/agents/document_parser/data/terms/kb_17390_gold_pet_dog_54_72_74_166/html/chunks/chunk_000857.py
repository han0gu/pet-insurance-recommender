from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 계약자가<br>제1회 보험료를 신용카드로 납입한 계약의 승낙을 거절하는 경우에는 신용카드</p><br><h1 '
 "id='24' style='font-size:16px'>의 매출을 취소하며 이자를 더하여 지급하지 않습니다.</h1><br><h1 "
 "id='25' style='font-size:16px'>제12조(특별약관의 무효)</h1><br><p id='26' "
 "data-category='paragraph' style='font-size:14px'>질<br>계약을 맺을 때에 계약에서 정한 "
 '반려동물의 나이에 미달되었거나'),
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
