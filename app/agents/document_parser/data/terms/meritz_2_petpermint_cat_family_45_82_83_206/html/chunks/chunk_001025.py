from langchain_core.documents import Document

chunk = Document(
    page_content=(". 지급률의 결정</h1><br><p id='45' data-category='paragraph' "
 "style='font-size:16px'>1) 한 팔의 3대 관절중 관절 하나에 기능장해가 생기고<br>다른 관절 하나에 기능장해가 "
 "발생한 경우 지급률은<br>각각 적용하여 합산한다.</p><footer id='46' "
 "style='font-size:14px'>192</footer><p id='47' data-category='paragraph' "
 "style='font-size:20px'>2) 1상지(팔과 손가락)의 장해 지급률은"),
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
