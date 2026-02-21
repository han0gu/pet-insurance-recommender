from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>약</p><br><p id='110' data-category='list' "
 "style='font-size:14px'>관<br>제1조(보험금의 지급사유)<br>\uf000 회사는 피보험자가 이 특별약관의 보험기간 "
 '중에 진단확정된 질병으로 병원 또는<br>의원(한방병원 또는 한의원을 포함합니다)에 입원하여 치료를 받은 경우에는 입<br>원기간 동안 '
 '보험증권에 기재된 반려동물을 수탁기관에 위탁함으로써 발생한 위<br>탁비용을 반려동물 위탁비용으로 보험수익자에게 지급합니다'),
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
