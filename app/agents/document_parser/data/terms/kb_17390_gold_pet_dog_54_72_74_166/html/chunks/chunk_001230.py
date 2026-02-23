from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>제21조(조사)</h1><br><p id='29' data-category='paragraph' "
 "style='font-size:16px'>회사는 보험의 목적에 대한 위험상태를 조사하기 위하여 보험기간 중 언제든지</p><br><p "
 "id='30' data-category='list' style='font-size:16px'>피보험자의 시설과 업무내용을 조사할 수 있고 "
 '필요한 경우에는 그의 개선을 피<br>보험자에게 요청할 수 있습니다.<br>\uf000 회사는 제1항에 따른 개선이 완료될 때까지'),
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
