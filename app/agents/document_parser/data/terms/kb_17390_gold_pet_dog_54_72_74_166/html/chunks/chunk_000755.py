from langchain_core.documents import Document

chunk = Document(
    page_content=("정의는, 이 특별약관의 다른 조항에서 달리 정의</p><br><h1 id='107' "
 "style='font-size:14px'>같습니다.</h1><br><h1 id='108' style='font-size:14px'>되지 "
 "않는 한 다음과</h1><br><h1 id='109' style='font-size:14px'>1"),
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
