from langchain_core.documents import Document

chunk = Document(
    page_content=('문서, 휴대전화 문자메시지 또는 이에 준하는 전자적 의사\n'
 '표시 등으로 알려드리고, 회사는 계약자의 재가입의사를 전\n'
 '화(음성녹음), 직접 방문 또는 전자적 의사표시, 통신판매\n'
 '계약의 경우 통신수단을 통해 확인합니다.\uf000 계약자는 제4항에 따른 재가입안내와 재가입여부 확인\n'
 '요청을 받은 경우 재가입 의사를 표시하여야 합니다.\n'
 '\uf000 제4항 및 제5항에도 불구하고, 회사가 계약자의 재가입\n'
 '의사를 확인하지 못한 경우(계약자와의 연락두절로 회사의\n'
 '안내가 계약자에게 도달하지 못한 경우 포함)에는 직전계약'),
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
