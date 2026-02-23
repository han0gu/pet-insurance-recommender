from langchain_core.documents import Document

chunk = Document(
    page_content=("id='42' data-category='list'></p><br><h1 id='43' style='font-size:16px'>8) "
 "약간의 운동장해</h1><br><p id='44' data-category='paragraph' "
 "style='font-size:16px'>머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경추)를 제외한 척추체(척</p><br><p "
 "id='45' data-category='paragraph' style='font-size:16px'>추뼈 몸통)에 골절 또는 탈구로 "
 '2개의 척추체(척추뼈 몸통)를'),
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
