from langchain_core.documents import Document

chunk = Document(
    page_content=("정한 계약 후 알릴 의무를 계약자 또는<br>피보험자의 고의 또는 중대한 과실로 이행하지 않았을<br>때</p><br><p id='73' "
 "data-category='paragraph' style='font-size:20px'>\uf000 제1항 제1호의 경우에도 불구하고 "
 "다음 중 하나에 해당<br>하는 경우에는 회사는 계약을 해지할 수 없습니다.</p><br><p id='74' "
 "data-category='list' style='font-size:16px'>① 회사가 최초계약 체결당시에 그 사실을 알았거나 "
 '과실<br>로 인하여 알지 못하였을'),
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
