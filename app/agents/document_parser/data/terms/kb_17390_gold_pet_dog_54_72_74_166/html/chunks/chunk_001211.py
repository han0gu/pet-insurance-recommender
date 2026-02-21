from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>보험의 목적에 아래와 같은 사실이 생긴 경우에는 계약자나 피보</p><br><p id='8' "
 "data-category='paragraph' style='font-size:14px'>험자는 지체없이 서면으로 회사에 알리고 보험증권에 "
 '확인을 받아야 합니다.<br>1'),
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
