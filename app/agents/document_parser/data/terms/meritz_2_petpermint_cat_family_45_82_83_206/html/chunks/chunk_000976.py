from langchain_core.documents import Document

chunk = Document(
    page_content=(". 손바닥 크기</h1><br><p id='73' data-category='paragraph' "
 "style='font-size:20px'>“손바닥 크기”라 함은 해당 환자의 손가락을 제외한 손<br>바닥의 크기를 말하며, 12세 "
 '이상의 성인에서는 8×10㎝<br>(1/2 크기는 40㎠, 1/4 크기는 20㎠), 6∼11세의 경우는<br>6×8㎝(1/2 크기는 '
 '24㎠, 1/4 크기는 12㎠), 6세 미만의<br>경우는 4×6㎝(1/2 크기는 12㎠, 1/4 크기는 6㎠)로 '
 "간<br>주한다.</p><h1 id='74'"),
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
