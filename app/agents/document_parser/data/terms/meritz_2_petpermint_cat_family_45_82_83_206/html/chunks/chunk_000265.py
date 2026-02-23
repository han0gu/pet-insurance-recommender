from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>83</footer><footer id='70' "
 "style='font-size:14px'>84</footer><h1 id='71' style='font-size:20px'>Ⅰ"),
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
