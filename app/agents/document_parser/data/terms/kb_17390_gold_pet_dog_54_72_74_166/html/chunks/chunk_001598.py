from langchain_core.documents import Document

chunk = Document(
    page_content=("무릎관절(슬관절),<br>발목관절(족관절)을 말한다.</p><p id='105' data-category='paragraph' "
 "style='font-size:20px'>- 148 -</p><p id='106' "
 "data-category='list'></p><header id='107' style='font-size:16px'>5) ‘한 다리의 "
 '발목 이상을 잃었을 때’라 함은 발목관절(족관절)부터(발<br>목관절 포함) 심장에 가까운 쪽에서 절단된 때를 말하며, '
 '무릎관절(슬관<br>절)의 상부에서 절단된 경우도'),
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
