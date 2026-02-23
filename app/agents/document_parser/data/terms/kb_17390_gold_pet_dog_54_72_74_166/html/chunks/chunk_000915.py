from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자, 피보험자 또는 보험수익자에게 손해를 가한 경우에는<br>그에 따른 손해를 배상할 책임을 집니다.<br>\uf000 회사가 보험금 '
 '지급여부 및 지급금액에 관하여 현저하게 공정을 잃은 합의로 보<br>험수익자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를 배상할 '
 "책임을<br>집니다.</p><br><p id='95' data-category='list'></p><br><table id='96' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>용 어 풀</td><td>이 현저하게"),
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
