from langchain_core.documents import Document

chunk = Document(
    page_content=('따라 보장을 받는 기간을 말합니다.</td></tr><tr><td>영업일</td><td>회사가 영업점에서 정상적으로 영업하는 날을 '
 '말하며, 토요일,‘관공서의 공휴일에 관한 규 정’에 따른 공휴일(대체공휴일 포함)과 근로 자의 날을 '
 "제외합니다.</td></tr></tbody></table><br><h1 id='6' style='font-size:20px'>\uf000 "
 "보험료 관련 용어</h1><br><table id='7'"),
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
