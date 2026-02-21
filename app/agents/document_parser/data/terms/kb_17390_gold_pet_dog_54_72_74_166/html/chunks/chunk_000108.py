from langchain_core.documents import Document

chunk = Document(
    page_content=('. 한편 위험이 증가<br>된 경우에는 보험료의 증액 및 정산금액의 추가납입을 요구할 수 있으며, 계약자<br>는 일시납 또는 잔여 '
 '보험료 납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초<br>과할 수 없음) 동안의 분납 중 선택하여 정산금액을 납입하여야 합니다'),
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
