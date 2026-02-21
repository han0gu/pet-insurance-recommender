from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 분류</h1><br><table id='53' style='font-size:16px'><thead><tr><td>장해의 "
 '분류</td><td>지급률</td></tr></thead><tbody><tr><td>1) 외모에 뚜렷한 추상(추한 모습)을 남긴 '
 '때</td><td>15</td></tr><tr><td>2) 외모에 약간의 추상(추한 모습)을 남긴 '
 "때</td><td>5</td></tr></tbody></table><h1 id='54' style='font-size:16px'>나"),
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
