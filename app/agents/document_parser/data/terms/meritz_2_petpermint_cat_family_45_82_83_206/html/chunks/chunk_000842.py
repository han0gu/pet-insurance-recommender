from langchain_core.documents import Document

chunk = Document(
    page_content=("부활(효력회복))</h1><br><p id='4' data-category='paragraph' "
 "style='font-size:20px'>회사는 이 특별약관의 부활(효력회복) 청약을 받은 경우에</p><footer id='5' "
 "style='font-size:14px'>167</footer><p id='6' data-category='paragraph' "
 "style='font-size:18px'>는 보험계약의 부활(효력회복)을 승낙한 경우에 한하여 보<br>통약관 제30조(보험료의 납입을 "
 '연체하여 해지된 계약의'),
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
