from langchain_core.documents import Document

chunk = Document(
    page_content=("id='48' data-category='paragraph' style='font-size:20px'>\uf000 회사는 제1항에 따른 "
 "계약자 명의변경 신청 및 계약의<br>특별부활(효력회복) 청약을 승낙합니다.</p><br><p id='49' "
 "data-category='paragraph' style='font-size:20px'>\uf000 회사는 제1항의 통지를 지정된 "
 '보험수익자에게 하여야<br>합니다'),
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
