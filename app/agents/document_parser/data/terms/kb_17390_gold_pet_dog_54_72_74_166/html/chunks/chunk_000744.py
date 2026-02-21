from langchain_core.documents import Document

chunk = Document(
    page_content=("매 사고시마다 지급합니다.</p><p id='84' data-category='list' "
 "style='font-size:14px'>제2조(보험금 지급에 관한 세부규정)<br>\uf000 제1조(보험금의 지급사유)의 깁스치료비는 "
 '같은 상해 또는 질병으로 인하여 깁스<br>치료를 2회 이상 받은 경우, 또는 동시에 서로 다른 신체부위에 깁스치료를 받은<br>경우에는 '
 '1회에 한하여 깁스치료비를 지급합니다.<br>\uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 '
 '합의하<br>지 못할 때는 보험수익자와 회사가 함께 제3자를'),
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
