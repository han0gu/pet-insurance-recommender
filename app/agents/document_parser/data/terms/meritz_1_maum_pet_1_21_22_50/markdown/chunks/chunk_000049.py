from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항에서 정한대로 계약자 또는 보험수익자가 변경내용을 알리지 않은 경우에는 계약\n'
 '- 자 또는 보험수익자가 회사에 알린 최종의 주소 또는 연락처로 등기우편 등 우편물에\n'
 '- 대한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 도달에 필요한 기간이 지\n'
 '- 난 때에 계약자 또는 보험수익자에게 도달된 것으로 봅니다.'),
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
