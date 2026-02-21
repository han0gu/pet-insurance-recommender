from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>2) 1상지(팔과 손가락)의 장해 지급률은 원칙적으로 각각<br>합산하되, 지급률은 60% "
 "한도로 한다.</p><h1 id='48' style='font-size:16px'>9"),
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
