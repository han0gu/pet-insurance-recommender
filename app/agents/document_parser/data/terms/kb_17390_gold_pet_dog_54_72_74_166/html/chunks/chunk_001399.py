from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>제4조(계약자의 알릴 의무)</p><br><p id='51' "
 "data-category='paragraph' style='font-size:14px'>\uf000 계약자가</p><br><p "
 "id='52' data-category='list'></p><br><p id='53' data-category='paragraph' "
 "style='font-size:14px'>제3조(약관교부의 특례) 제1항에 정한 방법으로 보험계약 안내자료를 수</p><br><p "
 "id='54'"),
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
