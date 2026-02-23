from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회사\n'
 '는 반려동물 위탁비용의 전부 또는 일부를 지급하지 않습니다.- \n'
 '126 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)\uf000 피보험자가 병원 또는 의원을 이전하여 입원한 경우에도 동일한 상해의 '
 '치료를 목\n'
 '적으로 2회 이상 입원한 경우에는 계속하여 입원한 것으로 보아 각 입원일수를 더\n'
 '합니다.\n'
 '\uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하'),
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
