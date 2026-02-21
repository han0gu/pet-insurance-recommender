from langchain_core.documents import Document

chunk = Document(
    page_content=('불구하고, 보건복지부에서 고시하는「건강보험 행위 급여․비급여 목록 관<br>및 급여 상대가치점수」의 개정에 따라 "창상봉합술 대상 '
 '수가코드"가 폐지 또는<br>변경되어 보험금 지급사유에 대해 판정이 불가능한 경우 폐지 또는 변경 직전의 관<br>련 법령에서 정한 '
 '"창상봉합술 대상 수가코드"를 따릅니다.<br>\uf000 제1항에도 불구하고, "건강보험 행위 급여․비급여 목록 및 급여 상대가치점수" '
 '개<br>정으로 급여 판정이 변경되더라도 제1조(보험금의 지급사유) 제1항의 지급사유 발 상해<br>생 당시의 "건강보험 행위 '
 '급여․비급여 목록'),
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
