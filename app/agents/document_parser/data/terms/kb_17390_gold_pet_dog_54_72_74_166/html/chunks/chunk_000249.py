from langchain_core.documents import Document

chunk = Document(
    page_content=('없는 경우를 포함합니다)<br>공<br>계약자는 해지된 날부터 3년 이내에 회사가 정한 절차에 따라 계약의 '
 '부활(효력회<br>통<br>복)을 청약할 수 있습니다'),
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
