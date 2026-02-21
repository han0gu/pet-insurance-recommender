from langchain_core.documents import Document

chunk = Document(
    page_content=(". 병</p><br><p id='214' data-category='paragraph' "
 "style='font-size:16px'>제3조(특별약관의 소멸)</p><br><p id='215' "
 "data-category='paragraph' style='font-size:16px'>피보험자가 사망하였을</p><br><p "
 "id='216' data-category='paragraph' style='font-size:14px'>반<br>해약환급금 "
 '산출방법서"에서 정하는 바에 따라 피보험자의 사망 당시 이 특별약관의 려<br>계약자적립액 및'),
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
