from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다)의 보험기간이 끝난 날의 다음 날을 말합니다.\n'
 '- 4. 갱신종료나이 : 사업방법서에서 정한 갱신형 계약의 갱신종료나이 계약해당일을 말\n'
 '- 합니다.\n'
 '⑦ (재가입형) 특별약관 재가입 관련 용어- 1. 최초계약 : 최초로 체결되는 계약을 말합니다.\n'
 '- 2 재가있게야 · OI ㅂ허이 사업방법 서에 서 저하 TULLOI 정차에 MPL 재가인되 게야으'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
