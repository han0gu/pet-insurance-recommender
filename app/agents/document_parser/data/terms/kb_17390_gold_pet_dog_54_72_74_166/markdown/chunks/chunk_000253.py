from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 장해분류표에 해당되지 않는 후유장해는 피보험자의 직업, 연령, 신분 또는 성별 동\n'
 '- 등에 관계없이 신체의 장해정도에 따라 장해분류표의 구분에 준하여 지급액을 결 물\n'
 '- 정합니다. 다만, 장해분류표의 각 장해분류별 최저 지급률 장해정도에 이르지 않\n'
 '- 는 후유장해에 대하여는 후유장해보험금을 지급하지 않습니다.\n'
 '- \uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의 제\n'
 '- 하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따 도'),
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
