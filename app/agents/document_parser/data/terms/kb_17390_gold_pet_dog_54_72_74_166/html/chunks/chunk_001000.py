from langchain_core.documents import Document

chunk = Document(
    page_content=('시행 당일 피보험자가 부담한 의료비 : 78만원<br>·반려동물의료비보험금 : 15만원<br>·지급금액 = {(78만원 - 15만원 - '
 "3만원) x 70%, 100만원} 중 적은 금액</p><br><h1 id='207' style='font-size:14px'>= "
 "42만원</h1><br><p id='208' data-category='list' "
 "style='font-size:14px'>예시②<br>·MRI/CT 시행 당일 피보험자가 부담한 의료비 : "
 '218만원<br>·반려동물의료비보험금 : 15만원<br>·지급금액 ='),
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
