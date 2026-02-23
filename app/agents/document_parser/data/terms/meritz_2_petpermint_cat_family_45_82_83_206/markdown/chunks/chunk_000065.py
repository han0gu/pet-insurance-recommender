from langchain_core.documents import Document

chunk = Document(
    page_content=('는 방법으로 위험 정도에 따라 특별보험료를 추가로 부\n'
 '가하는 방법을 말합니다.\uf000 회사는 계약의 청약을 받고, 제1회 보험료를 받은 경우\n'
 '에 건강진단을 받지 않는 계약은 청약일, 진단계약은 진단\n'
 '일(재진단의 경우에는 최종 진단일)부터 30일 이내에 승낙\n'
 '또는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니\n'
 '다. 그러나 30일 이내에 승낙 또는 거절의 통지가 없으면\n'
 '승낙된 것으로 봅니다.\n'
 '\uf000 회사가 제1회 보험료를 받고 승낙을 거절한 경우에는 거\n'
 '절통지와 함께 받은 금액을 계약자에게 돌려 드리며, 보험'),
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
