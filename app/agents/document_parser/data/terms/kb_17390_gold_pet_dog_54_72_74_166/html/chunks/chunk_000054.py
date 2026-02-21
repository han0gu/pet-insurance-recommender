from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>\uf000 회사는 제1항의 규정에 정한 지급기일내에 보험금을 지급하지 않았을 "
 '때(제2항의<br>규정에서 정한 지급예정일을 통지한 경우를 포함합니다)에는 그 다음날부터 지급<br>일까지의 기간에 대하여 "보험금을 '
 '지급할 때의 적립이율 계산"(【별표2】참조)<br>에서 정한 이율로 계산한 금액을 보험금에 더하여 지급합니다'),
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
