from langchain_core.documents import Document

chunk = Document(
    page_content=('2024년 4월 10일 2024년 7월 9일" data-coord="top-left:(150,343); '
 'bottom-right:(719,451)" /></figure><br><p id=\'223\' data-category=\'list\' '
 'style=\'font-size:16px\'>\uf000 제3항에서 "연간"이란 계약일로부터 매1년 단위로 도래하는 계약해당일 '
 '전일까지<br>기간을 의미합니다.<br>\uf000 반려동물(강아지) 일반조항 제22조(재가입) 제1항 및 제2항에 따라 재가입한 '
 '경<br>우 또는 반려동물(강아지) 일반조항 제22조(재가입)'),
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
