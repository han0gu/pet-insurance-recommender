from langchain_core.documents import Document

chunk = Document(
    page_content=(". 계약자, 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실</p><br><h1 id='40' "
 "style='font-size:14px'>【중과실(중대한 과실)】</h1><br><p id='41' "
 "data-category='paragraph' style='font-size:14px'>주의의무의 위반이 현저한 과실,「중대한 과실」, "
 '즉 현저한 부주의, 태만의 경우<br>로서 조금만 주의를 하였다면 충분히 피해의 발생을 막을 수 있었음에도 그 주의<br>조차 태만히 한 '
 '높은 강도의 주의의무 위반(이하'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
