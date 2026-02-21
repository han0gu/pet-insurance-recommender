from langchain_core.documents import Document

chunk = Document(
    page_content=("다음의 서류를 제출하고 보험금을 청구하여야 합니다.</h1><br><p id='81' data-category='list' "
 "style='font-size:14px'>1"),
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
