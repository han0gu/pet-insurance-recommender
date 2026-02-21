from langchain_core.documents import Document

chunk = Document(
    page_content=('제2천추 이하의 천<br>골, 미골, 좌골 포함), 빗장뼈(쇄골), 가슴뼈(흉골), 갈비뼈(늑골)를<br>말하며 이를 모두 동일한 부위로 '
 "본다.<br>2) ‘골반뼈의 뚜렷한 기형’이라 함은 아래의 경우 중 하나에 해당하는 때<br>를 말한다.</p><br><p id='73' "
 "data-category='list'></p><br><p id='74' data-category='paragraph' "
 "style='font-size:14px'>가) 천장관절 또는 치골문합부가 분리된 상태로 치유되었거나 좌골이<br>2.5cm이상 분리된 "
 '부정유합'),
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
