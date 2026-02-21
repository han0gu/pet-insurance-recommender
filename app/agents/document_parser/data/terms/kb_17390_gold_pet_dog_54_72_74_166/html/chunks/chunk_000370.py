from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 장해분류표의 각 장해분류별 최저 지급률 장해정도에 이르지 않</p><br><p id='31' "
 "data-category='paragraph' style='font-size:16px'>- 74 -</p><p id='32' "
 "data-category='list' style='font-size:16px'>는 후유장해에 대하여는 후유장해보험금을 지급하지 "
 '않습니다.<br>\uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의<br>하지 못할 때는 '
 '보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에'),
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
