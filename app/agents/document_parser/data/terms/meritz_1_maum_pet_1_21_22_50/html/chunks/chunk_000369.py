from langchain_core.documents import Document

chunk = Document(
    page_content=("법 제51조 제1항 제2호에 따른 장애인은 다음 각 호의 어느 하나에 해당하는 자로<br>한다.</p><br><p id='49' "
 "data-category='list' style='font-size:14px'>1.「장애인복지법」에 따른 장애인 및「장애아동 "
 '복지지원법」에 따른 장애아동 중<br>기획재정부령으로 정하는 사람<br>2.「국가유공자 등 예우 및 지원에 관한 법률」에 의한 상이자 및 '
 '이와 유사한 사람<br>으로서 근로능력이 없는 사람<br>3.「국민건강보험법 시행령」 별표2 제3호 라목1)부터10)까지 외의 부분 '
 '전단에'),
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
