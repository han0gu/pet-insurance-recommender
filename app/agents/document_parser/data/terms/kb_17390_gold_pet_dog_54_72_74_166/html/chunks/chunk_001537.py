from langchain_core.documents import Document

chunk = Document(
    page_content=("id='17' data-category='list'></p><br><p id='18' data-category='paragraph' "
 "style='font-size:14px'>라) 코의 1/2 이상 결손<br>2) 머리<br>가) 손바닥 크기 이상의 반흔(흉터) 및 "
 "모발결손<br>나) 머리뼈의 손바닥 크기 이상의 손상 및 결손<br>3) 목</p><br><p id='19' "
 "data-category='list'></p><br><p id='20' data-category='paragraph'"),
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
