from langchain_core.documents import Document

chunk = Document(
    page_content=('⇒ 모든 보험수익자가 장애인이 아니므로 이 특별약관을 적용할 수 없습니다.# <이 특별약관을 적용할 수 없는 사례 예시 2>전환대상계약의 '
 '보험수익자 1인은 비장애인이고 피보험자 2인 중 한명은 비장애인, 한명은 장애인인 경우\n'
 '⇒ 모든 피보험자가 장애인이 아니므로 이 특별약관을 적용할 수 없습니다.# <이 특별약관을 적용할 수 없는 사례 예시 3>전환대상계약의 '
 '피보험자는 비장애인이고 보험수익자가 법정상속인(장애인)인 경우\n'
 '⇒ 현재 법정상속인이 장애인이라고 하더라도 이 특별약관을 적용할 수 없습니다. 장애인전용보험으로'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
