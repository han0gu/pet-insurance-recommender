from langchain_core.documents import Document

chunk = Document(
    page_content=('해지 할 수 있습니다.<br>상<br>\uf000 회사는 계약자 또는 피보험자의 고의로 손해가 발생한 경우 이 특별약관을 '
 '해지<br>해<br>할 수 있습니다.<br>및<br>\uf000 제1항 및 제2항의 경우 회사는 계약자에게 이 특별약관의 해약환급금을 '
 "지급합니다.<br>질</p><h1 id='48' style='font-size:16px'>제25조(특별약관의 소멸)</h1><br><p "
 "id='49' data-category='paragraph' style='font-size:14px'>질</p><p id='50'"),
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
