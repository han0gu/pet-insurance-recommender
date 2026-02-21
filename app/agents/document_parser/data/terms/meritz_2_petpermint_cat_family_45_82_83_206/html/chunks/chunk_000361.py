from langchain_core.documents import Document

chunk = Document(
    page_content=("재가입 적용대상 특별약관(이하「재가<br>입 적용대상 특별약관」이라 합니다)이라 함은 아래의 특별</p><footer id='8' "
 "style='font-size:14px'>97</footer><h1 id='9' style='font-size:20px'>약관을 "
 "말합니다.</h1><br><h1 id='10' style='font-size:20px'>【재가입 적용대상 특별약관】</h1><br><p "
 "id='11' data-category='list' style='font-size:16px'>･ 펫퍼민트 반려묘 입원의료비보장"),
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
