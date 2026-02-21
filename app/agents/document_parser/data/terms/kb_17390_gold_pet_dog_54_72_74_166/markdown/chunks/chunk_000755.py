from langchain_core.documents import Document

chunk = Document(
    page_content=('차를 포함합니다.\n'
 '1. 이륜인 자동차에 측차를 붙인 자동차132 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)2. 조향장치의 조작방식, 동력전달방식 '
 '또는 원동기 냉각방식 등이 이륜의 자동차\n'
 '와 유사한 구조로 되어 있는 삼륜 또는 사륜의 자동차로서 승용자동차에 해당\n'
 '하지 않는 자동차- 3. 전동기를 이용한 동력발생장치를 사용하는 삼륜 또는 사륜의 자동차로서 승용\n'
 '- 자동차에 해당하지 않는 자동차\n'
 '- \uf000 제2항 및 제3항에서 자동차관리법(하위 법령, 규칙 포함) 및 도로교통법(하위 법'),
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
