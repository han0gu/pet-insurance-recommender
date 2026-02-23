from langchain_core.documents import Document

chunk = Document(
    page_content=("21만원</p><br><p id='72' data-category='paragraph' style='font-size:20px'>② 입원 "
 "중 수술을 한 경우(보상비율 70%)</p><br><p id='73' data-category='list' "
 "style='font-size:20px'>·피보험자가 부담한 수술당일 치료비 410만원<br>·보험금 지급금액</p><br><p "
 "id='74' data-category='list' style='font-size:20px'>= [(410만원-3만원)×70%, "
 '250만원] 중 적은금액<br>='),
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
