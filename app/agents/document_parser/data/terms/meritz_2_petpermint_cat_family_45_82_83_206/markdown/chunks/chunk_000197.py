from langchain_core.documents import Document

chunk = Document(
    page_content=('- 입 당시 정한 재가입 나이의 범위 내일 것\n'
 '- ② 재가입 전 계약의 보험료가 정상적으로 납입완료 되었\n'
 '- 을 것\n'
 '\uf000 이 재가입 적용대상 특별약관의 보험기간 종료 후 계약\n'
 '자가 재가입을 원하는 경우 계약자는 재가입 시점에서 회사\n'
 '가 판매하는 동일하거나 객관적이고 합리적인 범위내에서\n'
 '기존 계약내용에 상응한 반려동물보험 상품(보험업감독규정\n'
 '제1-2조(정의)에서 정한 장기손해보험에 한하며 이하「반려\n'
 '동물보험 상품」이라 합니다)으로 가입을 할 수 있으며, 회\n'
 '사는 이를 거절할 수 없습니다. 다만, 재가입 계약이 직전'),
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
