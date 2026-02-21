from langchain_core.documents import Document

chunk = Document(
    page_content=("id='52' style='font-size:20px'>제7조(보험금의 청구)</h1><br><p id='53' "
 "data-category='paragraph' style='font-size:20px'>\uf000 보험수익자는 다음의 서류를 제출하고 "
 "보험금을 청구하<br>여야 합니다.</p><br><p id='54' data-category='list' "
 "style='font-size:16px'>① 청구서(회사양식)<br>② 사고증명서(진단서, 진료비계산서, 사망진단서, 장해진단<br>서, "
 '입원치료확인서, 의사처방전(처방조제비) 등)<br>③'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
