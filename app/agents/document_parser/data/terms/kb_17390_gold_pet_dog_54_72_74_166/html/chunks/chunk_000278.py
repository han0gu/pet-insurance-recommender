from langchain_core.documents import Document

chunk = Document(
    page_content=("id='102' data-category='list'></p><br><p id='103' data-category='paragraph' "
 "style='font-size:14px'>다.<br>\uf000 해약환급금의 지급사유가 발생한 경우 계약자는 회사에 해약환급금을 "
 "청구하여야</p><br><p id='104' data-category='list'></p><p id='105' "
 "data-category='paragraph' style='font-size:16px'>- 68 -</p><p id='106'"),
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
