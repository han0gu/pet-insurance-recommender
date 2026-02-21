from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자는 제4항에 따른 재가입안내와 재가입여부 확인<br>요청을 받은 경우 재가입 의사를 표시하여야 합니다.<br>\uf000 제4항 및 '
 '제5항에도 불구하고, 회사가 계약자의 재가입<br>의사를 확인하지 못한 경우(계약자와의 연락두절로 회사의<br>안내가 계약자에게 도달하지 '
 '못한 경우 포함)에는 직전계약<br>과 동일한 조건으로 보험계약을 연장합니다'),
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
