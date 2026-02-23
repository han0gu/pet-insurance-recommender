from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>- 127 -</p><br><p id='119' data-category='paragraph' "
 "style='font-size:14px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 127</p><p id='120' "
 "data-category='list' style='font-size:14px'>끝났을 때에도 퇴원하기 전까지의 계속중인 입원기간에 "
 '대하여는 제1조(보험금의<br>지급사유) 제3항에 따라 반려동물 위탁비용을 계속 지급합니다.<br>\uf000 피보험자가 정당한 이유없이 '
 '입원기간 중'),
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
