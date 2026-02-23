from langchain_core.documents import Document

chunk = Document(
    page_content=("받지 않은 날)</p><br><p id='31' data-category='list' "
 "style='font-size:20px'>·피보험자가 부담한 치료비 13만원<br>·보험금 지급금액</p><br><p id='32' "
 "data-category='list' style='font-size:20px'>= [(13만원 - 3만원)×50%, 10만원] 중 "
 "적은금액<br>= 5만원</p><br><p id='33' data-category='paragraph' "
 "style='font-size:20px'>② 입원 중 MRI,CT 및 내시경처치를"),
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
