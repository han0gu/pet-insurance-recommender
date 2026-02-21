from langchain_core.documents import Document

chunk = Document(
    page_content=('말한다.<br>마) “약간의 뇌전증 발작”이라 함은 월 1회 이상의 중증발작 또는 월 2<br>회 이상의 경증발작이 연 6개월 이상의 '
 '기간에 걸쳐 발생하는 상태<br>를 말한다.<br>바) “중증발작”이라 함은 전신경련을 동반하는 발작으로써 신체의 균<br>형을 유지하지 '
 '못하고 쓰러지는 발작 또는 의식장해가 3분 이상 지<br>속되는 발작을 말한다.<br>사) “경증발작”이라 함은 운동장해가 발생하나 '
 '스스로 신체의 균형을<br>유지할 수 있는 발작 또는 3분 이내에 정상으로 회복되는 발작을 말<br>한다.</p><br><p'),
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
