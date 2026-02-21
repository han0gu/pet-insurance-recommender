from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 같습니다)을 체결할 때 반려동<br>물의 건강상태가 회사가 정한 기준에 적합하지 않은 경우<br>또는 보험계약을 체결한 후 계약 '
 '전 알릴 의무 위반의 효과<br>등으로 보장을 제한할 경우 보험계약자(이하 「계약자」라<br>합니다)의 청약과 보험회사의 승낙으로 '
 '보험계약(이하 「계<br>약」이라 합니다)에 부가하여 이루어집니다.<br>\uf000 제1항에 따라 이 특약을 부가할 때 반려동물의 과거 '
 '병<br>력과 수의학적으로 또는 경험통계적으로 인과관계가 유의성<br>있게 확인된 경우 등과 같이 회사가 정한 기준에 따라 '
 '직접<br>관련이'),
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
