from langchain_core.documents import Document

chunk = Document(
    page_content=("보험료는 아래에 기재된 납입기일까지 납입하여야 합니다.</p><br><p id='23' data-category='paragraph' "
 "style='font-size:14px'>( )회 분납: 제 1회: 계약의 청약일 (총 보험료의 ( )% 해당액)<br>제( )회: 년 "
 "월 일 (총 보험료의 ( )% 해당액)</p><br><p id='24' data-category='list' "
 "style='font-size:14px'>② 보험기간이 시작된 후라도 제1항의 제1회 나눠 내는 보험료를 납입하기 전에 생긴 "
 '사<br>고는 보상하여 드리지'),
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
