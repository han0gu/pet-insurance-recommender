from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 61</p><br><p id='167' "
 "data-category='paragraph' style='font-size:18px'>- 61 -</p><header id='168' "
 "style='font-size:14px'>\uf000 회사가 제2항에 따라 일부보장 제외 조건을 붙여 승낙하였더라도 청약일로부터 "
 '5년<br>(갱신형 계약의 경우에는 최초 계약의 청약일로부터 5년)이 지나는 동안 보장이 제<br>외되는 질병으로 추가 진단(단순 '
 '건강검진 제외)'),
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
