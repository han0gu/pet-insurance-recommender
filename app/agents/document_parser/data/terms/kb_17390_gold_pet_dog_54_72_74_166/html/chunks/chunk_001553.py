from langchain_core.documents import Document

chunk = Document(
    page_content=('. 통약<br>6) 심한 운동장해란 다음 중 어느 하나에 해당하는 경우를 말한다.<br>관<br>가) 척추체(척추뼈 몸통)에 골절 또는 '
 "탈구로 4개 이상의 척추체(척추뼈</p><br><p id='40' data-category='paragraph' "
 "style='font-size:16px'>몸통)를 유합(아물어 붙음) 또는 고정한 상태<br>나) 머리뼈(두개골), 제1경추, 제2경추를 "
 '모두 유합 또는 고정한 상태<br>7) 뚜렷한 운동장해란 다음 중 어느 하나에 해당하는 경우를 말한다.<br>가) 척추체(척추뼈 몸통)에 '
 '골절 또는'),
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
