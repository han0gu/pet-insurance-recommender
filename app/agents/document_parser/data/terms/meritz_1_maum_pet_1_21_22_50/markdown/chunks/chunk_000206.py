from langchain_core.documents import Document

chunk = Document(
    page_content=('입영수증에 보험료 공제대상임이 표시된 보험의 보험료를 말한다.2. 모든 피보험자 또는 모든 보험수익자가「소득세법 시행령 '
 '제107조(장애인의 범위) 제\n'
 '1항」에서 규정한 장애인인 보험【「소득세법 시행령 제107조(장애인의 범위) 제1항」에서 규정한 장애인】① 법 제51조 제1항 제2호에 '
 '따른 장애인은 다음 각 호의 어느 하나에 해당하는 자로\n'
 '한다.- 1.「장애인복지법」에 따른 장애인 및「장애아동 복지지원법」에 따른 장애아동 중\n'
 '- 기획재정부령으로 정하는 사람'),
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
