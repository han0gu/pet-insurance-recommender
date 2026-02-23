from langchain_core.documents import Document

chunk = Document(
    page_content=("수 있을 정도의</p><br><p id='175' data-category='list' "
 "style='font-size:16px'>시력상태<br>5) 안구(눈동자) 운동장해의 판정은 질병의 진단 또는 외상 후 1년 "
 '이상이<br>지난 뒤 그 장해 정도를 평가한다.<br>6) ‘안구(눈동자)의 뚜렷한 운동장해’ 라 함은 아래의 두 경우 중 '
 "하나에<br>해당하는 경우를 말한다.</p><br><p id='176' data-category='paragraph' "
 "style='font-size:16px'>가) 한 눈의 안구(눈동자)의"),
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
