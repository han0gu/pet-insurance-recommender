from langchain_core.documents import Document

chunk = Document(
    page_content=("일반 보통인이라면 그 같은 일을 하지 않을<br>정도로 현저하게 공정성을 잃은 것을 말합니다.</p><h1 id='56' "
 "style='font-size:20px'>제46조(개인정보보호)</h1><br><p id='57' "
 "data-category='paragraph' style='font-size:20px'>\uf000 회사는 이 계약과 관련된 개인정보를 "
 '이 계약의 체결,<br>유지, 보험금 지급 등을 위하여「개인정보 보호법」,「신용<br>정보의 이용 및 보호에 관한 법률」등 관계 법령에 '
 '정한 경<br>우를 제외하고 계약자, 피보험자 또는'),
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
