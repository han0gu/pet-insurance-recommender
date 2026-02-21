from langchain_core.documents import Document

chunk = Document(
    page_content=("계약을 해지하고 보험금을 지급하지<br>않을 수 있습니다.</p><h1 id='21' "
 "style='font-size:20px'>제16조(상해보험계약 후 알릴 의무)</h1><br><p id='22' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약자 또는 피보험자는 보험기간 중에 "
 '피보험자에게 다<br>음 각 호의 변경이 발생한 경우에는 우편, 전화, 방문 등의<br>방법으로 지체없이 회사에 알려야 '
 "합니다.</p><br><p id='23' data-category='paragraph'"),
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
