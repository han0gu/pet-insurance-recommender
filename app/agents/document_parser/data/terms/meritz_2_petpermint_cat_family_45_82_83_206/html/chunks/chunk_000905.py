from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:18px'>【별표2】</p><h1 id='33' "
 "style='font-size:20px'>장해분류표</h1><h1 id='34' style='font-size:18px'>\uf000 "
 "총칙</h1><h1 id='35' style='font-size:18px'>1"),
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
