from langchain_core.documents import Document

chunk = Document(
    page_content=("소멸)</p><br><p id='36' data-category='paragraph' "
 "style='font-size:16px'>피보험자가</p><br><p id='37' data-category='paragraph' "
 'style=\'font-size:16px\'>사망하였을 경우에는 이 특별약관 계약도 소멸되며 회사는 "보험료 및</p><br><p '
 "id='38' data-category='paragraph' style='font-size:16px'>해약환급금 "
 '산출방법서"에서 정하는 바에 따라 피보험자의 사망 당시 이'),
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
