from langchain_core.documents import Document

chunk = Document(
    page_content=(". 전환대상계약의 피보험자는 비장애인이고 보험수익자가 법정상속인(장애인)인 경우</p><br><p id='59' "
 "data-category='paragraph' style='font-size:14px'>: 현재 법정상속인이 장애인이라고 하더라도 이 "
 '특별약관을 적용할 수 없습니다'),
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
