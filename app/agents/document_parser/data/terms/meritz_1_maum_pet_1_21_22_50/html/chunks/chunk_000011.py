from langchain_core.documents import Document

chunk = Document(
    page_content=(". 지급금과 이자율 관련 용어</h1><br><p id='16' data-category='list' "
 "style='font-size:14px'>가. 연단위 복리: 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 "
 '이자를<br>원금에 더한 금액을 다음 1년의 원금으로 하는 이자 계산방법을 말합니다.<br>나'),
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
