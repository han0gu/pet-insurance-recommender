from langchain_core.documents import Document

chunk = Document(
    page_content=("체결 및</p><br><p id='44' data-category='paragraph' "
 "style='font-size:14px'>효력)</p><br><p id='45' data-category='list' "
 "style='font-size:14px'>\uf000 이 특별약관은 보통약관(다른 특별약관이 부가된 경우에는 그 특별약관도 "
 '포함합<br>니다'),
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
