from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>\uf000 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게<br>유리하게 "
 '해석합니다.<br>\uf000 회사는 보험금을 지급하지 않는 사유 등 계약자나 피보<br>험자에게 불리하거나 부담을 주는 내용은 확대하여 '
 "해석하<br>지 않습니다.</p><p id='40' data-category='paragraph' "
 "style='font-size:20px'>제43조(설명서 교부 및 보험안내자료 등의 효력)</p><br><p id='41' "
 "data-category='paragraph'"),
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
