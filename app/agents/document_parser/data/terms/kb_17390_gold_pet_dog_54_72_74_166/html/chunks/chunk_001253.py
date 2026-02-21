from langchain_core.documents import Document

chunk = Document(
    page_content=("2</p><br><p id='72' data-category='paragraph' style='font-size:14px'>회 이상 "
 '입원한 경우 이를 1회 입원으로 보아 각 입원일수를 더합니다.<br>\uf000 피보험자가 보장개시일 이후 입원하여 치료를 받던 중 '
 '보험기간이 끝났을 때에도<br>퇴원하기 전까지의 계속 중인 입원에 대하여는 제1조(보험금의 지급사유) 제3항에<br>따라 반려동물 '
 '위탁비용을 계속 지급합니다.<br>\uf000 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회사<br>는 '
 '반려동물 위탁비용의 전부'),
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
