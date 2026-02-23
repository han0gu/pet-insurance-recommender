from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>제39조(분쟁의 조정)</h1><br><p id='25' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약에 관하여 분쟁이 있는 경우 분쟁 "
 '당사자 또는 기타<br>이해관계인과 회사는 금융감독원장에게 조정을 신청할 수<br>있으며, 분쟁조정 과정에서 계약자는 관계 법령이 '
 '정하는<br>바에 따라 회사가 기록 및 유지･관리하는 자료의 열람(사본<br>의 제공 또는 청취를 포함한다)을 요구할 수 '
 '있습니다.<br>\uf000 회사는 일반금융소비자인 계약자가'),
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
