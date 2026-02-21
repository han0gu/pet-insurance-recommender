from langchain_core.documents import Document

chunk = Document(
    page_content=('상병의 진료를 위하여 최초로 내원(입원<br>물<br>을 포함합니다)한 날을 말합니다)로 합니다.<br>\uf000 제1항의 '
 '"골절철심제거술"은 의료법 제3조(의료기관)에서 정한 국내의 병원 또는<br>국외의 의료관련법에서 정한 의료기관의 의사(치과의사 제외) '
 '면허를 가진 자(이<br>제<br>하 "의사"라 합니다)에 의하여 "골절철심제거술"이 필요하다고 인정한 경우로서<br>도<br>"의사"의 '
 '관리하에 의료법 제3조(의료기관) 제2항에서 규정한 국내의 병원 및 의<br>성<br>원에서 행한 의료행위에 '
 '한합니다.<br>특<br>\uf000'),
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
