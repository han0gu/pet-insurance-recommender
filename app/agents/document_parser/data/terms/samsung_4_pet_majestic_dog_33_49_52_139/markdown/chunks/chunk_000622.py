from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>[핵연료물질]사용된 연료를 포함합니다.\n'
 '[핵연료물질에 의하여 오염된 물질]\n'
 '원자핵 분열 생성물을 포함합니다.6. 반려견을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으# 로 '
 '이용함으로써 발생한 손해7. 수의사의 치료상의 과오로 생긴 손해, 수의사 자격이 없는 자의 치료행위로 인한\n'
 '손해(수의사의 소견 및 처방에 의한 경우도 동일) 및 그로 인하여 가중된 손해\n'
 '8. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
