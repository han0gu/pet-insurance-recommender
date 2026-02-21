from langchain_core.documents import Document

chunk = Document(
    page_content=('중 적은 금액이 100만<br>원인 경우 중도인출 가능액은 80만원(100만원의 80%)이<br>며, 보험계약대출금(원금과 이자의 합계가 '
 "30만원이라고<br>가정)이 있는 경우 중도인출 가능액은 50만원(80만원-30<br>만원)입니다.</p><footer id='22' "
 "style='font-size:14px'>78</footer><h1 id='23' style='font-size:20px'>제7관 분쟁의 "
 "조정 등</h1><h1 id='24' style='font-size:16px'>제39조(분쟁의 조정)</h1><br><p"),
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
