from langchain_core.documents import Document

chunk = Document(
    page_content=('. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변<br>4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성, '
 "그 밖의 유해한<br>특성 또는 이들의 특성에 의한 사고</p><p id='34' data-category='paragraph' "
 "style='font-size:14px'>【핵연료물질】사용된 연료를 포함합니다.<br>【핵연료물질에 의하여 오염된 물질】원자핵 분열 "
 "생성물을 포함합니다.</p><p id='35' data-category='list' style='font-size:14px'>5"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
