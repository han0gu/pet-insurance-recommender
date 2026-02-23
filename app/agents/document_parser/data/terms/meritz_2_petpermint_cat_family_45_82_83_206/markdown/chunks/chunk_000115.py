from langchain_core.documents import Document

chunk = Document(
    page_content=('분 절차에 따라 회사는 채권자에게 해약환급금을 지급하\n'
 '게 됩니다.제6관 계약의 해지 및 해약환급금 등제32조(계약자의 임의해지 및 피보험자의 서면동의 철회)\uf000 계약자는 계약이 '
 '소멸하기 전에는 언제든지 계약을 해지\n'
 '할 수 있으며, 이 경우 회사는 제35조(해약환급금) 제1항에\n'
 '따른 해약환급금을 계약자에게 지급합니다.\n'
 '\uf000 제22조(계약의 무효)에 따라 사망을 보험금 지급사유로\n'
 '하는 계약에서 서면으로 동의를 한 피보험자는 계약의 효력\n'
 '이 유지되는 기간에는 언제든지 서면동의를 장래를 향하여'),
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
