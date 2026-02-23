from langchain_core.documents import Document

chunk = Document(
    page_content=('- 6) 심한 운동장해란 다음 중 어느 하나에 해당하는 경우를 말한다.\n'
 '- 가) 척추체(척추뼈 몸통)에 골절 또는 탈구 등으로 4개 이상의 척추체(척추뼈\n'
 '- 몸통)를 유합(아물어 붙음) 또는 고정한 상태\n'
 '- 나) 머리뼈(두개골), 제1경추, 제2경추를 모두 유합 또는 고정한 상태\n'
 '# 7) 뚜렷한 운동장해란 다음 중 어느 하나에 해당하는 경우를 말한다.- 가) 척추체(척추뼈 몸통)에 골절 또는 탈구 등으로 3개의 '
 '척추체(척추뼈 몸통)\n'
 '- 를 유합(아물어 붙음) 또는 고정한 상태'),
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
