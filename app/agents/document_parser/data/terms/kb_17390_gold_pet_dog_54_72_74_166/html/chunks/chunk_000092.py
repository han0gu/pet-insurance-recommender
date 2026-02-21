from langchain_core.documents import Document

chunk = Document(
    page_content=("전문의를 둘 것</p><p id='112' data-category='paragraph' "
 "style='font-size:18px'>제15조(상해보험계약 후 알릴 의무) 통약<br>\uf000 계약자 또는 피보험자는 보험기간 "
 "중에 피보험자에게 다음 각 호의 변경이 발생한</p><br><p id='113' data-category='paragraph' "
 "style='font-size:16px'>경우에는 우편, 전화, 방문 등의 방법으로 지체없이</p><br><p id='114' "
 "data-category='paragraph'"),
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
