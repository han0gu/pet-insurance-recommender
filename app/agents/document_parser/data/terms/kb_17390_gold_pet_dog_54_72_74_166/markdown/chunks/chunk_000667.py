from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 피보험자가 피해자에게 지급할 책임을 지는 법률상의 손해배상금 질\n'
 '7. 피보험자의 불법행위 또는 폭력행위로 인한 배상책임\n'
 '2. 계약자 또는 피보험자가 지출한 아래의 비용 병\n'
 '8. 티끌, 먼지, 석면, 분진 또는 소음으로 생긴 손해에 대한 배상책임\n'
 '가. 피보험자가 제12조(손해방지의무) 제1항 제1호의 손해의 방지 또는 경감\n'
 '9. 전자파, 전자장(EMF)으로 생긴 손해에 대한 배상책임\n'
 '을 위하여 지출한 필요 또는 유익하였던 비용\n'
 '10. 벌과금 및 징벌적 손해에 대한 배상책임'),
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
