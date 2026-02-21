from langchain_core.documents import Document

chunk = Document(
    page_content=('장해가 생기고 다른 발가락에 장해가 발생한 경우, 지급률<br>통<br>은 각각 적용하여 합산한다.<br>8) 발가락 관절의 운동범위 '
 '측정은 장해평가시점의 ｢산업재해보상보험법 시 사항<br>행규칙｣ 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한 평균 '
 "운</p><br><table id='150' style='font-size:20px'><thead><tr><td>동가능영역을 "
 '기준으로</td><td>정상각도 및 측정방법 등을 따른다.</td></tr></thead><tbody><tr><td>부 가 설 명'),
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
