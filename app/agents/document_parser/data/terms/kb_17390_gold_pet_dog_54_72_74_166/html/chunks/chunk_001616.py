from langchain_core.documents import Document

chunk = Document(
    page_content=('. 장해판정기준<br>1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것이 기능장해의 원인<br>이 되는 때에는 그 내고정물 등이 '
 '제거된 후에 장해를 평가한다. 단, 제<br>거가 불가능한 경우에는 고정물 등이 있는 상태에서 장해를 평가한다.<br>2) 관절을 '
 '사용하지 않아 발생한 일시적인 기능장해(예를 들면 캐스트로 환<br>부를 고정시켰기 때문에 치유 후의 관절에 기능장해가 발생한 '
 '경우)는<br>장해로 평가하지 않는다.<br>3) 손가락에는 첫째 손가락에 2개의 손가락관절이 있다'),
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
