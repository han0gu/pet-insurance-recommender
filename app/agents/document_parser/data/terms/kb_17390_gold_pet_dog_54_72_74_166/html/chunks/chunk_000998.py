from langchain_core.documents import Document

chunk = Document(
    page_content=('시 주요치료보험금의 계산<br>[주요치료보험금 산출방식]<br>{(피보험자가 부담한 당일 의료비 - 반려동물의료비보험금 - '
 "자기부담금)</p><br><h1 id='202' style='font-size:14px'>X 보상비율}과 치료구분별 보상한도액 중 적은 "
 "금액</h1><br><p id='203' data-category='list' style='font-size:14px'>[MRI/CT "
 "시행시 지급금액 예시]<br>·보상비율 70%, 자기부담금 3만원, 기본형Ⅱ 가입 기준</p><br><p id='204'"),
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
