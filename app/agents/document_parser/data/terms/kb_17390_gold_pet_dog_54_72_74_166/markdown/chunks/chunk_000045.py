from langchain_core.documents import Document

chunk = Document(
    page_content=('| 예) 보험금: 6천만원, 보험금 | 지급일자: 2024년 4월 10일 일때 보험금을 일시 <table><thead><tr><td>에 '
 '받지않고 3년 동안 매년 동일한</td><td>금액으로 나누어 지급받는 '
 '경우</td></tr></thead><tbody><tr><td>지급일</td><td>지급 금액</td></tr><tr><td>2024년 '
 '4월 10일</td><td>2천만원</td></tr><tr><td>2025년 4월 10일</td><td>2천만원 X (1+ '
 '평균공시이율)</td></tr></tbody></table>'),
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
