from langchain_core.documents import Document

chunk = Document(
    page_content=("범위)</h1><br><p id='9' data-category='paragraph' "
 "style='font-size:20px'>\uf000 이 특별약관에서 피보험자라 함은 아래에 정한 보험증권<br>에 기재된 피보험자 및 "
 "그 가족을 말합니다.</p><br><p id='10' data-category='list' style='font-size:20px'>① "
 '보험증권에 기재된 피보험자(이하「피보험자 본인」이<br>라 합니다)<br>② 피보험자 본인의 가족관계등록상 또는 주민등록상에<br>기재된 '
 '배우자(이하「배우자」라 합니다)<br>③ 피보험자'),
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
