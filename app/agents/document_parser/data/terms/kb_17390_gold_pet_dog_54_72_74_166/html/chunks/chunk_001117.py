from langchain_core.documents import Document

chunk = Document(
    page_content=('. 청구서(회사 양식)<br>2. 국가동물 등록한 경우에는 동물등록증 또는 등록번호<br>3. 국가동물 미등록한 경우에는 가입동물의 사진 '
 '2매(얼굴전면, 측면전신사진)를<br>회사에 제출하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금<br>을 '
 '지급합니다.<br>4. 동물장묘업자가 제공하는 장례확인서(동물장묘업소 등록번호, 업소명 및 주<br>소, 전화번호, 서비스 대상 동물의 '
 '종류, 품종, 나이, 장례서비스 이용일자,<br>화장서비스 등의 서비스 이용내역, 비용 등 포함)<br>5'),
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
