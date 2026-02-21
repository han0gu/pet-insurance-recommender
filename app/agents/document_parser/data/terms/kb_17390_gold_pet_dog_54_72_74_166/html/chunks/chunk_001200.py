from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사가 보상한 금액이 피보 특<br>험자가 입은 손해의 일부인 경우에는 피보험자의 권리를 침해하지 않는 범위 내 '
 "약</p><br><p id='223' data-category='paragraph' style='font-size:20px'>- 123 "
 "-</p><br><p id='224' data-category='paragraph' "
 "style='font-size:14px'>질</p><p id='225' data-category='paragraph' "
 "style='font-size:16px'>KB 금쪽같은"),
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
