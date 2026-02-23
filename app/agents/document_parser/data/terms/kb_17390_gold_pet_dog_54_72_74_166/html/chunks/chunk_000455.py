from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>외모특정상해(머리, 목)수술비</p><br><p id='149' "
 "data-category='paragraph' style='font-size:14px'>병</p><h1 id='150' "
 "style='font-size:16px'>제1조(보험금의 지급사유)</h1><br><p id='151' "
 "data-category='paragraph' style='font-size:14px'>회사는 피보험자가 이 특별약관의 보험기간 중에 "
 '상해의 직접결과로써 "외모특정상<br>및<br>해"로 진단확정되고 그'),
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
