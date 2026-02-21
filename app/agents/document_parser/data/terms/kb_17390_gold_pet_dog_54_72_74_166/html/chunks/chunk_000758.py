from langchain_core.documents import Document

chunk = Document(
    page_content=("id='111' data-category='paragraph' style='font-size:14px'>100 KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01)</p><br><table id='112' "
 "style='font-size:14px'><thead><tr><td>용 "
 '어</td><td></td></tr></thead><tbody><tr><td>반려동물</td><td>정 의 보험증권에 기재된 반려동물을 '
 '말하며, 이 계약에서 가입 가능 한 반려동물은 대한민국 내에서 피보험자와 거주를 함께하고 있 는 개(犬)를 말합니다'),
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
