from langchain_core.documents import Document

chunk = Document(
    page_content=("천자 : 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것</h1><p id='143' "
 "data-category='paragraph' style='font-size:16px'>제4조(특별약관의 소멸)<br>피보험자가 "
 '사망하였을 경우에는 이 특별약관 계약도 소멸되며 회사는 "보험료 및<br>해약환급금 산출방법서"에서 정하는 바에 따라 피보험자의 사망 '
 "당시 이 특별약관의 상해</p><br><p id='144' data-category='paragraph' "
 "style='font-size:16px'>계약자적립액 및"),
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
