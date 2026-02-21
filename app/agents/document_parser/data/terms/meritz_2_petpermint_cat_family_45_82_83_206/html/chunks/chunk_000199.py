from langchain_core.documents import Document

chunk = Document(
    page_content=('제1항에 따라 계약이 해지된 경우에는 제35조(해약환급<br>금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.</p><p '
 "id='65' data-category='paragraph' style='font-size:16px'>제30조(보험료의 납입을 연체하여 "
 "해지된 계약의 부활(효력<br>회복))</p><br><p id='66' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제29조(보험료의 납입이 연체되는 경우 납입최고(독촉)<br>와 계약의 해지)에 "
 '따라 계약이 해지되었으나'),
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
