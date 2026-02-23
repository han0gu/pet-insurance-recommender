from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자는 회사가 정당한 사유 없이 제1항의 요구를 따르지 않는 경우 해당 계약을<br>해지할 수 있습니다.<br>\uf000 제1항 및 '
 '제3항에 따라 계약이 해지된 경우 회사는 제34조(해약환급금) 제5항에<br>따른 해약환급금을 계약자에게 지급합니다.<br>\uf000 '
 "계약자는 제1항에 따른 제척기간에도 불구하고 민법 등 관계 법령에서 정하는 바</p><br><p id='78' "
 "data-category='list'></p><br><p id='79' data-category='paragraph' "
 "style='font-size:14px'>에 따라"),
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
