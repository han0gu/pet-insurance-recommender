from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변\n'
 '- 4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성, 그 밖의 유해한\n'
 '- 특성 또는 이들의 특성에 의한 사고\n'
 '【핵연료물질】사용된 연료를 포함합니다.\n'
 '【핵연료물질에 의하여 오염된 물질】원자핵 분열 생성물을 포함합니다.- 5. 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '- 6. 피보험자의 피고용인이 피보험자의 업무에 종사중에 입은 신체의 장해로 인한 배상책임'),
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
