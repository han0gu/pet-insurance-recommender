from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자 또는 피보험자가 고의 또는 중대한 과실로 제1항<br>각 호의 변경사실을 회사에 알리지 않았을 경우 변경후 요<br>율이 변경전 '
 '요율보다 높을 때에는 회사는 그 변경사실을<br>안 날로부터 1개월 이내에 계약자 또는 피보험자에게 제4항<br>에 의해 보장됨을 '
 "통보하고 이에 따라 보험금을 지급합니<br>다.</p><footer id='44' "
 "style='font-size:14px'>60</footer><p id='45' data-category='paragraph' "
 "style='font-size:20px'>【중대한"),
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
