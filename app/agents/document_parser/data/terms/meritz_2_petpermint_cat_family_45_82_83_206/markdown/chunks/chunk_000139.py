from langchain_core.documents import Document

chunk = Document(
    page_content=('계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 보\n'
 '험회사 및 보험관련단체 등에 개인정보를 제공할 수 있습니다.\n'
 '\uf000 회사는 계약과 관련된 개인정보를 안전하게 관리하여야\n'
 '합니다.# 제47조(준거법)이 계약은 대한민국 법에 따라 규율되고 해석되며, 약관에\n'
 '서 정하지 않은 사항은 금융소비자보호에 관한 법률, 상법,81# 민법 등 관계 법령을 따릅니다.제48조(예금보험에 의한 지급보장)회사가 '
 '파산 등으로 인하여 보험금 등을 지급하지 못할 경\n'
 '우에는 예금자보호법에서 정하는 바에 따라 그 지급을 보장'),
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
