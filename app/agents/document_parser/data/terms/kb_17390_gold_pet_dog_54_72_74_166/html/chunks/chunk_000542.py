from langchain_core.documents import Document

chunk = Document(
    page_content=(". 선천적 기형 및 이에 근거한 병상</p><h1 id='5' style='font-size:14px'>제6조(수술의 "
 "정의와</h1><br><p id='6' data-category='paragraph' "
 "style='font-size:14px'>장소)</p><br><p id='7' data-category='list' "
 'style=\'font-size:14px\'>\uf000 이 특별약관에 있어서 "수술"이라 함은 병원 또는 의원의 의사, 치과의사 '
 '면허를<br>가진 자(이하 "의사"라 합니다)에 의하여 치료가 필요하다고 인정한 경우로서'),
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
