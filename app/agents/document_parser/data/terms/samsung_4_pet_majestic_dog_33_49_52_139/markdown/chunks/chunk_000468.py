from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변\n'
 '- 4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성 또는 그 밖의\n'
 '- 유해한 특성 또는 이들 특성에 의한 사고\n'
 '- 5. 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '# <용어풀이># [핵연료물질]# 사용된 연료를 포함합니다.# [핵연료물질에 의하여 오염된 물질]# 원자핵 분열 생성물을 포함합니다.- '
 '6. 피보험자의 질병, 심신상실 또는 정신질환으로 인한 손해\n'
 '- 7. 최초계약의 보험계약일 이전에 이미 감염 또는 발병한 상해 및 질병'),
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
