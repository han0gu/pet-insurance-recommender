from langchain_core.documents import Document

chunk = Document(
    page_content=(". 선천적 기형 및 이에 근거한 병상</p><br><h1 id='249' style='font-size:14px'>제4조(수술의 정의와 "
 "장소)</h1><br><h1 id='250' style='font-size:14px'>\uf000 이 특별약관에 있어서 "
 '"수술"이라</h1><br><p id=\'251\' data-category=\'paragraph\' '
 "style='font-size:14px'>함은 병원 또는 의원의 의사, 치과의사 면허를</p><br><p id='252' "
 "data-category='list'"),
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
