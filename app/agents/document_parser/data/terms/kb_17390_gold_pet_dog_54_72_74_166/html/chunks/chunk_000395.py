from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>도로 합니다.</p><br><h1 id='56' "
 "style='font-size:14px'>제3조(특별약관의 소멸)</h1><br><h1 id='57' "
 "style='font-size:14px'>피보험자가</h1><br><p id='58' data-category='paragraph' "
 'style=\'font-size:14px\'>사망하였을 경우에는 이 특별약관 계약도 소멸되며 회사는 "보험료 및</p><br><p '
 "id='59' data-category='paragraph'"),
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
