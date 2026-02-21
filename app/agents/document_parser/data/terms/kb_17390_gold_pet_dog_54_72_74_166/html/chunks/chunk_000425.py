from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>관</p><p id='103' data-category='paragraph' "
 'style=\'font-size:14px\'>은 제외합니다.<br>\uf000 제1항의 "성형수술"은 피보험자가 사고발생시점에 만15세 '
 "미만일 경우 부득이 사</p><br><p id='104' data-category='paragraph' "
 "style='font-size:14px'>고일로부터 2년이 지난 후에 성형수술이 가능하다는 진단을 받은 경우에는 그 "
 "진</p><br><h1 id='105'"),
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
