from langchain_core.documents import Document

chunk = Document(
    page_content=('반려동물(강아지) 일반조항<br>제17조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복)) 및 제18조(강<br>제집행 등으로 '
 "인하여 해지된 특별약관의 특별부활(효력회복))에 따라 이 특별약관의<br>부활(효력회복)을 취급합니다.</p><br><p id='128' "
 "data-category='paragraph' style='font-size:16px'>청약을 받은 경우에는 보험계약의 "
 "부활(효력회복)</p><p id='129' data-category='paragraph' style='font-size:18px'>- "
 '139'),
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
