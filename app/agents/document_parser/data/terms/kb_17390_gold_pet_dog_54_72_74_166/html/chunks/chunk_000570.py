from langchain_core.documents import Document

chunk = Document(
    page_content=('병원이나<br>상<br>의원 또는 국외의 의료관련법에서 정한 의료기관에서 발급한 것이어야 합니다.<br>해</p><br><p '
 "id='52' data-category='paragraph' style='font-size:14px'>질</p><h1 id='53' "
 "style='font-size:16px'>제5조(특별약관의 소멸)</h1><br><p id='54' "
 "data-category='paragraph' style='font-size:14px'>피보험자가 사망하였을 경우에는 이 특별약관의 "
 '계약도 소멸되며 회사는 "보험료 및'),
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
