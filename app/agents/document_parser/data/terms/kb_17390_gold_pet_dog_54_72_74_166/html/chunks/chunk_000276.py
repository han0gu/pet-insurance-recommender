from langchain_core.documents import Document

chunk = Document(
    page_content=("있습니다.</p><br><p id='97' data-category='paragraph' "
 "style='font-size:14px'>\uf000 제1항에 따라 해지하지 않은 계약은 파산선고 후 3개월이 지난 때에는 그 "
 '효력을<br>잃습니다.<br>\uf000 제1항에 따라 계약이 해지되거나 제2항에 따라 계약이 효력을 잃는 경우에 '
 "회사는<br>제34조(해약환급금) 제1항에 의한 해약환급금을 계약자에게 지급합니다.</p><br><p id='98' "
 "data-category='list'></p><br><h1 id='99'"),
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
