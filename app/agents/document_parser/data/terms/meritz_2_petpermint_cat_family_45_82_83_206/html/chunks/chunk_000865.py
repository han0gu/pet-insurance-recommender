from langchain_core.documents import Document

chunk = Document(
    page_content=('방광의 양성 신생물 기타</td></tr><tr><td>AGB003</td><td>방광의 악성 '
 '신생물</td></tr><tr><td></td><td></td></tr><tr><td>AGC003</td><td>기타 방광의 신생물 '
 "(양성 또는 악성이 불확 실한)</td></tr></tbody></table><footer id='15' "
 "style='font-size:14px'>170</footer><table id='16' "
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
