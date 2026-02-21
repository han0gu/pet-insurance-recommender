from langchain_core.documents import Document

chunk = Document(
    page_content=('. 청구서(회사 양식)<br>2. 국가동물 등록한 경우에는 동물등록증 또는 등록번호<br>3. 국가동물 미등록한 경우에는 가입동물의 사진 '
 '2매(얼굴전면, 측면전신사진)를<br>회사에 제출하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금 상<br>을 지급합니다. '
 '해<br>및<br>4. 사망을 확인할 수 있는 서류(동물폐사확인서, 동물화장증명서 등)<br>질<br>5'),
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
