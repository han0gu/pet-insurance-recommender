from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그 밖에 금융상품의 유형별로 대통령령으로 정하는 자<br>향후 관련법령이 개정된 경우 개정된 내용을 적용합니다.</p><br><p '
 "id='196' data-category='paragraph' style='font-size:14px'>※</p><br><p "
 "id='197' data-category='paragraph' style='font-size:14px'>\uf000</p><br><p "
 "id='198' data-category='paragraph' style='font-size:14px'>제1항에도 불구하고 청약한 날부터 "
 '30일(단, 만'),
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
