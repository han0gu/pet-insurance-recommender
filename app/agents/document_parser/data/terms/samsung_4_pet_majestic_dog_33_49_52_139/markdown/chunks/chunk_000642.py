from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변\n'
 '- 4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성 또는 그 밖의\n'
 '- 유해한 특성 또는 이들 특성에 의한 사고\n'
 '- 5. 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '<용어풀이># [핵연료물질]| 사용된 연료를 | 포함합니다. |\n'
 '| --- | --- |\n'
 '| [핵연료물질에 | 의하여 오염된 물질] |\n'
 '# 원자핵 분열 생성물을 포함합니다.② 회사는 피보험자가 다음에 열거한 배상책임을 부담함으로써 입은 손해를 보상하지 않'),
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
