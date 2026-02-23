from langchain_core.documents import Document

chunk = Document(
    page_content=("id='117' data-category='paragraph' style='font-size:14px'>제2조(특별면책(회사가 보험금을 "
 "지급하지 않는)조건의 내용)</p><br><p id='118' data-category='paragraph' "
 "style='font-size:14px'>\uf000 이 특별약관에서 정한 회사가 보험금을 지급하지 않는 기간 중에 "
 "회사가</p><br><p id='119' data-category='paragraph' style='font-size:14px'>지정한 "
 "질</p><br><p id='120'"),
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
