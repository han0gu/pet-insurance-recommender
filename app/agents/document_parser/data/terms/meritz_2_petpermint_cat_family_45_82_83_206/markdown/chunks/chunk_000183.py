from langchain_core.documents import Document

chunk = Document(
    page_content=('사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습니\n'
 '다.94\uf000 제1항에 따라 계약이 취소된 경우에는 회사는 이미 납입\n'
 '한 보험료를 계약자에게 돌려 드립니다.# 제11조(보험계약의 성립)\uf000 이 특별약관은 계약자의 청약과 회사의 승낙으로 이루어\n'
 '집니다.\n'
 '\uf000 회사는 피보험자 또는 반려동물이 계약에 적합하지 않은\n'
 '경우에는 승낙을 거절하거나 별도의 조건(보험가입금액 제\n'
 '한, 일부보장 제외, 보험금 삭감, 보험료 할증 등)을 붙여\n'
 '승낙할 수 있습니다.# 【보험가입금액 제한】반려동물이 가입을 할 수 있는 최대 보험가입금액을 제'),
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
