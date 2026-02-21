from langchain_core.documents import Document

chunk = Document(
    page_content=('피부 신생물 (양성 또는 악성이 불확실 한)</td></tr><tr><td>GAA001</td><td>외이도염 '
 '(세균성)</td></tr><tr><td>GAA002 GAA003</td><td>외이도염 (말라세지아) 외이도염 '
 "(알러지성)</td></tr></tbody></table><footer id='17' "
 "style='font-size:14px'>171</footer><table id='18' "
 "style='font-size:14px'><thead><tr><td>구"),
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
