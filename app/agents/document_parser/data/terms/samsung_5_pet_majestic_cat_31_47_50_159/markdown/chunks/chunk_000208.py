from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 특별약관은 청약일, 진단계약은 진단일(재진단의 경우에는 최종진단일)부터 30일\n'
 '- 이내에 승낙 또는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니다. 그러나 30일\n'
 '- 이내에 승낙 또는 거절의 통지가 없으면 승낙된 것으로 봅니다.\n'
 '- ⑤ 회사가 제1회 보험료를 받고 승낙을 거절한 경우에는 거절통지와 함께 받은 금액을\n'
 '- 계약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 평균공시이율+1%를 연단위\n'
 '- 복리로 계산한 금액을 더하여 지급합니다. 다만, 회사는 계약자가 제1회 보험료를 신'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
