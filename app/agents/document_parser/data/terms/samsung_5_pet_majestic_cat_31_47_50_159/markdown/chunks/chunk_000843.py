from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4) 뇌전증\n'
 '가) "뇌전증" 이라 함은 돌발적 뇌파이상을 나타내는 뇌질환으로 발작(경련,\n'
 '의식장해 등)을 반복하는 것을 말한다.\n'
 '나) 뇌전증 발작의 빈도 및 양상은 지속적인 항뇌전증제(항경련제) 약물로도 조\n'
 '절되지 않는 뇌전증을 말하며, 진료기록에 기재되어 객관적으로 확인되는\n'
 '뇌전증 발작의 빈도 및 양상을 기준으로 한다.\n'
 '다) "심한 뇌전증 발작" 이라 함은 월 8회 이상의 중증발작이 연 6개월 이상의\n'
 '기간에 걸쳐 발생하고, 발작할 때 유발된 호흡장애, 흡인성 폐렴, 심한 탈'),
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
