from langchain_core.documents import Document

chunk = Document(
    page_content=(". 선천적 기형 및 이에 근거한 병상</p><h1 id='169' style='font-size:14px'>제4조(외모특정상해의 정의 "
 "및</h1><br><p id='170' data-category='paragraph' "
 "style='font-size:14px'>진단확정)</p><br><p id='171' data-category='list' "
 'style=\'font-size:14px\'>\uf000 이 특별약관에 있어서 "외모특정상해"라 함은 【별표3】(외모특정상해 '
 '분류표)에<br>서 정한 상해를 말합니다.<br>\uf000 제1항의 "외모특정상해"의'),
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
