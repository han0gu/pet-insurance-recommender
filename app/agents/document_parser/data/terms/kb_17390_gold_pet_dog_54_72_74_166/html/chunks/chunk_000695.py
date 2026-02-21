from langchain_core.documents import Document

chunk = Document(
    page_content=(". 반<br>려</p><br><p id='255' data-category='paragraph' "
 "style='font-size:16px'>제3조(환경성질환의 정의 및 진단확정)</p><br><p id='256' "
 "data-category='paragraph' style='font-size:16px'>\uf000 이 특별약관에 있어서 "
 '"환경성질환"이라</p><br><p id=\'257\' data-category=\'list\' '
 'style=\'font-size:14px\'>(환경성질환 분류표)에서 정한 질환을 말합니다.<br>\uf000 제1항의 "환경성질환"의'),
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
