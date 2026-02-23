from langchain_core.documents import Document

chunk = Document(
    page_content=("id='51' data-category='list' style='font-size:14px'>\uf000 같은 상해로 두 가지 이상의 "
 '후유장해가 생긴 경우에는 후유장해 지급률을 합산<br>하여 지급합니다'),
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
