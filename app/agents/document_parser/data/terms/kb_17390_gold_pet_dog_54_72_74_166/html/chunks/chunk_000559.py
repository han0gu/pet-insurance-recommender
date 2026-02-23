from langchain_core.documents import Document

chunk = Document(
    page_content=("data-category='list' style='font-size:16px'>\uf000 제1조(보험금의 지급사유)의 골절부목치료비는 "
 '같은 상해를 직접적인 원인으로<br>골절 진단후, 다수의 부목치료를 받거나 동시에 서로 다른 신체부위에 부목치<br>료를 받은 경우에는 '
 '1회에 한하여 골절부목치료비를 지급합니다.<br>\uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 '
 '합<br>의하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에<br>따를 수 있습니다'),
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
