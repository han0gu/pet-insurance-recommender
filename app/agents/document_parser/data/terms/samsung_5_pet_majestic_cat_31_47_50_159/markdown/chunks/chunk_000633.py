from langchain_core.documents import Document

chunk = Document(
    page_content=('에 자기공명영상(MRI)을 사용하는 촬영 의료행위를 말합니다.\n'
 '② 이 특별약관에 있어서 「컴퓨터단층촬영(CT)」 이라 함은 제1조(보험금의 지급사유)에\n'
 '서 정한 수의사에 의하여 진단 및 치료가 필요하다고 인정된 경우로서 수의사의 관리\n'
 '하에 컴퓨터단층촬영(CT)을 사용하는 촬영 의료행위를 말합니다.- \n'
 '<용어풀이># [자기공명영상(MRI)]강한 자기장 내에서 인체에 고주파를 전사해서 반향 되는 전자기파를 측정하는 영상진단법\n'
 '[컴퓨터단층촬영(CT)]'),
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
