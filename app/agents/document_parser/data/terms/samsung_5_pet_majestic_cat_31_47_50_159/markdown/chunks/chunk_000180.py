from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 피상속인의 형제자매 ④ 피상속인의 4촌 이내의 방계혈족\n'
 '[직계비속]\n'
 '자기로부터 직계로 이어져 내려가는 혈족. 자녀, 손자, 증손 등\n'
 '[직계존속]\n'
 '조상으로부터 직계로 내려와 자기에 이르는 사이의 혈족. 부모, 조부모 등\n'
 '[방계혈족]\n'
 '자기의 형제자매와 형제자매의 직계비속, 직계존속의 형제자매 및 그 형제자매의 직계비속# 제14조 (대표자의 지정)① 계약자 또는 '
 '보험수익자가 2명 이상인 경우에는 각 대표자를 1명 지정하여야 합니다.\n'
 '이 경우 그 대표자는 각각 다른 계약자 또는 보험수익자를 대리하는 것으로 합니다.'),
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
