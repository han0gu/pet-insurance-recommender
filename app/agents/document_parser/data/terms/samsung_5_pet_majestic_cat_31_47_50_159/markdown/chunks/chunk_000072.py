from langchain_core.documents import Document

chunk = Document(
    page_content=('- 습니다.\n'
 '# <용어풀이># [보험가입금액 제한]피보험자가 가입을 할 수 있는 최대 보험가입금액을 제한하는 방법을 말합니다.\n'
 '[일부보장 제외]\n'
 '일반적인 경우보다 위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 특정 질병 또는 특정 신\n'
 '체 부위를 보장에서 제외하는 방법을 말합니다.\n'
 '[보험금 삭감]\n'
 '일반적인 경우보다 위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 보험 가입 후 기간이 경\n'
 '과함에 따라 위험의 크기 및 정도가 점차 감소하는 위험에 대해 적용하여 보험 가입 후 일정기간'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
