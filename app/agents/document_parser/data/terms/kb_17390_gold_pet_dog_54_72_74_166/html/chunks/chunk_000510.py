from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>합니다.</h1><p id='245' data-category='list' "
 "style='font-size:14px'>제 2조(보험금 지급에 관한 세부규정)<br>\uf000 제1조(보험금의 지급사유)의 "
 '골절수술비는 같은 상해로 두 종류 이상의 골절수술<br>을 받거나 같은 종류의 수술을 2회 이상 받은 경우에는 하나의 골절수술비만 '
 '지급<br>합니다.<br>\uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의<br>하지 못할 때는 '
 '보험수익자와 회사가 함께 제3자를 정하고 그'),
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
