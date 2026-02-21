from langchain_core.documents import Document

chunk = Document(
    page_content=('. 재가입 전 계약의 보험료가 정상적으로 납입완료 되었을 것<br>\uf000 이 계약의 보험기간 종료 후 계약자가 재가입을 원하는 경우 '
 "계약자는 재가입 시</p><br><p id='102' data-category='paragraph' "
 "style='font-size:14px'>점에서 회사가 판매하는 동일하거나 객관적이고 합리적인 범위내에서 기존 계약</p><p "
 "id='103' data-category='paragraph' style='font-size:14px'>108 KB 금쪽같은 "
 '펫보험(강아지)(무배당)(26.01)</p><br><h1'),
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
