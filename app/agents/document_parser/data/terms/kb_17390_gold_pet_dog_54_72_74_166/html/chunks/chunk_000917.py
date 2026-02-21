from langchain_core.documents import Document

chunk = Document(
    page_content=("id='99' data-category='paragraph' style='font-size:14px'>계약이 다음 각 호의 조건을 "
 "충족하고 계약자가 제4항에 따라 재가입 의사를 표시</p><br><p id='100' data-category='paragraph' "
 "style='font-size:14px'>한 때에는 제11조(보험계약의 성립) 및 보통약관 제1절 일반조항 제20조(약관 교<br>부 및 "
 '설명 의무 등)를 준용하여 회사가 정한 절차에 따라 계약자는 기존 계약에<br>이어 재가입할 수 있으며, 이 경우 회사는 기존계약의'),
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
