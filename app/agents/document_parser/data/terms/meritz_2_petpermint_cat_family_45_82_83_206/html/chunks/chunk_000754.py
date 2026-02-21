from langchain_core.documents import Document

chunk = Document(
    page_content=("21만원</p><br><p id='78' data-category='paragraph' style='font-size:20px'>② 입원 "
 '중 MRI,CT 및 내시경처치를 받은 날의 경우(보<br>상비율 70% 가입, 연간 첫번째 MRI,CT 및 '
 "내시경처<br>치)</p><br><p id='79' data-category='list' "
 "style='font-size:20px'>·피보험자가 부담한 치료비 103만원<br>·보험금 지급금액</p><br><p id='80' "
 "data-category='list'"),
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
