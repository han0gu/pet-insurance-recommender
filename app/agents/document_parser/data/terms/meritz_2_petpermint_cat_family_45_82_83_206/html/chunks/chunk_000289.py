from langchain_core.documents import Document

chunk = Document(
    page_content=("포함), 동물병원 진료기록부, X-ray 등 방사선</p><footer id='15' "
 "style='font-size:14px'>87</footer><p id='16' data-category='paragraph' "
 "style='font-size:20px'>촬영을 하는 경우 해당 사진(촬영일자 및 시간 필수),<br>기타 지불 증빙서류 "
 "등)</p><br><p id='17' data-category='list' style='font-size:16px'>③ "
 '신분증(주민등록증이나 운전면허증 등 사진이 붙은 정<br>부기관발행 신분증,'),
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
