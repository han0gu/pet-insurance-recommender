from langchain_core.documents import Document

chunk = Document(
    page_content=('. 뚜렷한 위험의 변경 또는 증가와 관련된 제17조(계약 후 알릴 의무)에서 정한<br>계약 후 알릴 의무를 계약자 또는 피보험자의 고의 '
 "또는 중대한 과실로 이행</p><br><p id='17' data-category='paragraph' "
 "style='font-size:14px'>하지 않았을 때<br>\uf000 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 "
 "경우에는 회사는 계</p><p id='18' data-category='list' style='font-size:14px'>\uf000 "
 '제1항에 의한 계약의 해지가 손해발생 전에'),
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
