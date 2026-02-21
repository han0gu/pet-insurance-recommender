from langchain_core.documents import Document

chunk = Document(
    page_content=("파산선고와 해지)</h1><br><p id='4' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사가 파산의 선고를 받은 때에는 계약자는 계약을 해<br>지할 수 "
 '있습니다.<br>\uf000 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후<br>3개월이 지난 때에는 그 효력을 '
 '잃습니다.<br>\uf000 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에<br>따라 계약이 효력을 잃는 경우에 회사는 '
 '제35조(해약환급<br>금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.</p><h1'),
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
