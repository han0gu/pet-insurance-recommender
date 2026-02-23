from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 부목(Splint Cast)치료는 제외합니다.<br>\uf000 제1항의 "부목(Splint Cast)치료"라 함은 석고붕대 '
 '또는 섬유유리붕대(Fiberglass<br>Cast)를 고정할 부분의 일측면 또는 양측면에 착용시키고 대주는 치료법을 '
 "말합니다.</p><br><p id='88' data-category='paragraph' "
 "style='font-size:14px'>제4조(특별약관의 소멸)</p><br><h1 id='89' "
 "style='font-size:14px'>피보험자가</h1><br><p id='90'"),
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
