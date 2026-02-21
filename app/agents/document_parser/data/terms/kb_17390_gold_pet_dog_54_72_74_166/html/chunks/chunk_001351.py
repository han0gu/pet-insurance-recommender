from langchain_core.documents import Document

chunk = Document(
    page_content=("id='228' data-category='paragraph' style='font-size:14px'>병</p><p id='229' "
 "data-category='paragraph' style='font-size:14px'>\uf000 회사는 제6조(보험금의 청구)에 정한 "
 '서류를 접수한 때에는 접수증을 드리고 그<br>서류를 접수한 날부터 3영업일 이내에 이 특별약관의 보험금을 드립니다'),
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
