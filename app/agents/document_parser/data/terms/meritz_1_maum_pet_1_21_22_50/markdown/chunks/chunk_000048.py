from langchain_core.documents import Document

chunk = Document(
    page_content=('금액을 다음 1년의 원금으로 하는 이자 계산방법을 말합니다.\n'
 '원금 100원, 이자율 연 10%를 가정할 때- - 1년 후 원리금 : 100원 + (100원×10%) = 110원\n'
 '- - 2년 후 원리금 : 110원 + (110원×10%) = 121원\n'
 '# 제12조(주소변경통지)- ① 계약자(보험수익자가 계약자와 다른 경우 보험수익자를 포함합니다)는 주소 또는 연락\n'
 '- 처가 변경된 경우에는 지체없이 그 변경내용을 회사에 알려야 합니다.\n'
 '- ② 제1항에서 정한대로 계약자 또는 보험수익자가 변경내용을 알리지 않은 경우에는 계약'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
