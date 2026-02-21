from langchain_core.documents import Document

chunk = Document(
    page_content=('몸통)를 포함하여 측정하며, 생리적 정상만곡을 고려하여 평가한다.\n'
 '나) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통)의 압박률, 골절의 부위 등을\n'
 '기준으로 판정한다. 척추체(척추뼈 몸통)의 압박률은 인접 상 · 하부[인접\n'
 '상 + 하부 척추체(척추뼈 몸통)에 진구성 골절이 있거나, 다발성 척추골절이\n'
 '있는 경우에는 골절된 척추와 가장 인접한 상 + 하부] 정상 척추체(척추뼈\n'
 '몸통)의 전방 높이의 평균에 대한 골절된 척추체(척추뼈 몸통) 전방 높이의\n'
 '감소비를 압박률로 정한다.'),
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
