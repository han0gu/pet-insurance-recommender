from langchain_core.documents import Document

chunk = Document(
    page_content=('경우를 포함합니다)에는 그 다음날부터 지급<br>일까지의 기간에 대하여 "보험금을 지급할 때의 적립이율 계산"(【별표2】참조)<br>에서 '
 '정한 이율로 계산한 금액을 보험금에 더하여 지급합니다'),
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
