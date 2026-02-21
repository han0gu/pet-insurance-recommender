from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>- 68 -</p><p id='106' data-category='list'></p><p "
 "id='107' data-category='paragraph' style='font-size:16px'>하며, 회사는 청구를 접수한 "
 '날부터 3영업일 이내에 해약환급금을 지급합니다'),
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
