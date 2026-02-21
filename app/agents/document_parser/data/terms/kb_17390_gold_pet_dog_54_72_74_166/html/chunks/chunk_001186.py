from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제1항 제3호의 경우에는 소송비용(중재 또는 조정에 관한 비용 포함) 및 변호</p><br><p id='205' "
 "data-category='paragraph' style='font-size:16px'>사비용과 회사의 동의를 받지 않은 행위에 따라 "
 "증가된 손해</p><p id='206' data-category='list' "
 "style='font-size:16px'>제13조(손해배상청구에 대한 회사의 해결)<br>\uf000 피보험자가 피해자에게 손해배상책임을 "
 '지는 사고가 생긴 때에는 피해자는 이 특<br>별약관에 따라 회사가 피보험자에게'),
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
