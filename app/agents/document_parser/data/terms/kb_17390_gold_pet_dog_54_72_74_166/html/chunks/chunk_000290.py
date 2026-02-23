from langchain_core.documents import Document

chunk = Document(
    page_content=("= 70만원</p><p id='116' data-category='paragraph' "
 "style='font-size:16px'>제37조(배당금의 지급)</p><br><p id='117' "
 "data-category='paragraph' style='font-size:16px'>회사는 이 보험에 대하여 계약자에게 배당금을 "
 "지급하지 않습니다.</p><br><p id='118' data-category='paragraph' "
 "style='font-size:14px'>보</p><br><p id='119'"),
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
