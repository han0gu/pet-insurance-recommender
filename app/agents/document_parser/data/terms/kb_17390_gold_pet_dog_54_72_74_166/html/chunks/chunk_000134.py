from langchain_core.documents import Document

chunk = Document(
    page_content=("계약유지 의사를 포기하여 만기일 이전에 계약관계를 청산하는 것</p><p id='157' data-category='list' "
 "style='font-size:16px'>제17조(사기에 의한 계약)<br>\uf000 계약자 또는 피보험자가 대리진단, 약물사용을 "
 '수단으로 진단절차를 통과하거나<br>진단서 위·변조 또는 청약일 이전에 암 또는 사람면역결핍바이러스병(HIV) 감염<br>의 진단 확정을 '
 '받은 후 이를 숨기고 가입하는 등 사기에 의하여 계약이 성립되었 특별<br>음을 회사가 증명하는 경우에는 계약일부터 5년 이내(사기사실을 '
 '안 날부터'),
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
