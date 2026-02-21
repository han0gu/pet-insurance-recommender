from langchain_core.documents import Document

chunk = Document(
    page_content=('장해 지급률을 준용한다.<br>8) 상기 장해항목에 해당되지 않는 장기간의 간병이 필요한 만성질환(만성간<br>질환, 만성폐쇄성폐질환 '
 "등)은 장해의 평가 대상으로 인정하지 않는다.</p><br><p id='177' data-category='list'></p><br><p "
 "id='178' data-category='list'></p><h1 id='179' "
 "style='font-size:16px'>13.</h1><br><h1 id='180' "
 "style='font-size:16px'>신경계․정신행동 장해</h1><br><table"),
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
