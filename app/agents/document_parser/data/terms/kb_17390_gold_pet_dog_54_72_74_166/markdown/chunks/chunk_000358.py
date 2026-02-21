from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다)한 날을 말합니다)로 합니다.\n'
 '- \uf000 제2항에서 "연간"이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전까지\n'
 '- 기간을 의미합니다.\n'
 '# 제2조(보험금 지급에 관한세부규정)보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지\n'
 '못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있\n'
 '습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하'),
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
