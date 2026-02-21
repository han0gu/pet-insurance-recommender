from langchain_core.documents import Document

chunk = Document(
    page_content=('. 청약서의 기재사항을 변경하고자 할 때 또는 변경이 생겼음을 알았을 때<br>2. 이 계약에서 보장하는 위험과 동일한 위험을 보장하는 '
 '계약을 다른 보험자와<br>체결하고자 할 때 또는 이와 같은 계약이 있음을 알았을 때<br>3. 반려동물을 양도할 때</p><br><p '
 "id='2' data-category='paragraph' style='font-size:14px'>4"),
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
