from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장애인전용보험전환</p><br><p id='66' data-category='list' "
 "style='font-size:14px'>제1조(적용범위)</p><br><p id='67' data-category='paragraph' "
 "style='font-size:14px'>\uf000 이 특별약관은 회사가 정한 방법에 따라 계약자가 청약(請約)하고 회사가 "
 '승낙(承<br>諾)함으로써 다음 각 호의 조건을 모두 만족하는 보험계약(이하 "전환대상계약"이<br>라 합니다)에 대하여 '
 '장애인전용보험으로 전환을 청약하는 경우에 적용합니다.</p><p'),
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
