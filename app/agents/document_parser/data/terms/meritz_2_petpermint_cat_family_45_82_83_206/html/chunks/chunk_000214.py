from langchain_core.documents import Document

chunk = Document(
    page_content=('계약의 효력<br>이 유지되는 기간에는 언제든지 서면동의를 장래를 향하여<br>철회할 수 있으며, 서면동의 철회로 계약이 해지되어 '
 "회사가</p><footer id='81' style='font-size:14px'>75</footer><p id='82' "
 "data-category='paragraph' style='font-size:20px'>지급하여야 할 해약환급금이 있을 때에는 "
 "제35조(해약환급<br>금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.</p><h1 id='83'"),
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
