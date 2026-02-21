from langchain_core.documents import Document

chunk = Document(
    page_content=('. 국가동물 미등록한 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를<br>회사에 제출하고 가입동물이 보험에 가입한 동물과 '
 "동일함을 확인 후 보험금<br>을 지급합니다.</p><h1 id='173' "
 "style='font-size:14px'>제8조(보험금의</h1><br><h1 id='174' "
 "style='font-size:14px'>지급절차)</h1><br><p id='175' data-category='list' "
 "style='font-size:14px'>\uf000 회사는 제7조(보험금의 청구)에서 정한 서류를 접수한 때에는"),
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
