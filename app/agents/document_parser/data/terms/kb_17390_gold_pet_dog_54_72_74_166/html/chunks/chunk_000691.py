from langchain_core.documents import Document

chunk = Document(
    page_content=('(120일) (180일) (120일)" data-coord="top-left:(812,251); '
 'bottom-right:(1438,453)" /></figure><br><p id=\'254\' data-category=\'list\' '
 "style='font-size:14px'>\uf000 피보험자가 질병에 대한 보장개시일 이후 입원하여 치료를 받던 중 보험기간이 "
 '병<br>끝났을 때에도 퇴원하기 전까지의 계속중인 입원기간에 대하여는 제1조(보험금의<br>지급사유) 제2항에 따라 환경성질환입원일당을 '
 '계속 지급합니다.<br>\uf000 피보험자가 정당한 이유없이'),
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
