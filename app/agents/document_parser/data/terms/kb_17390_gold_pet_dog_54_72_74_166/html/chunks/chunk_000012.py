from langchain_core.documents import Document

chunk = Document(
    page_content=('. 지급금과</td><td>이자율 관련 용어</td></tr></thead><tbody><tr><td>용 연단위</td><td>어 정 의 '
 '회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를 원금에 더한 금액을 다음 1년의 원금으로 하는 이자 계산방법을 '
 '말합니다'),
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
