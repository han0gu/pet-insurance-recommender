from langchain_core.documents import Document

chunk = Document(
    page_content=('과 기본계약 적립부분 해약환급금 중 적은 금액이 100만\n'
 '원인 경우 중도인출 가능액은 80만원(100만원의 80%)이\n'
 '며, 보험계약대출금(원금과 이자의 합계가 30만원이라고\n'
 '가정)이 있는 경우 중도인출 가능액은 50만원(80만원-30\n'
 '만원)입니다.78# 제7관 분쟁의 조정 등# 제39조(분쟁의 조정)\uf000 계약에 관하여 분쟁이 있는 경우 분쟁 당사자 또는 기타\n'
 '이해관계인과 회사는 금융감독원장에게 조정을 신청할 수\n'
 '있으며, 분쟁조정 과정에서 계약자는 관계 법령이 정하는\n'
 '바에 따라 회사가 기록 및 유지･관리하는 자료의 열람(사본'),
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
