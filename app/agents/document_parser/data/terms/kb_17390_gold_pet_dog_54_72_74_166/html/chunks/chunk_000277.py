from langchain_core.documents import Document

chunk = Document(
    page_content=("id='98' data-category='list'></p><br><h1 id='99' "
 "style='font-size:14px'>제34조(해약환급금)</h1><br><p id='100' "
 'data-category=\'paragraph\' style=\'font-size:14px\'>해약환급금은"보험료 및 해약환급금 '
 '산출방법서"에 따라 계산합니</p><br><h1 id=\'101\' style=\'font-size:14px\'>\uf000 이 약관에 '
 "따른</h1><br><p id='102' data-category='list'></p><br><p"),
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
