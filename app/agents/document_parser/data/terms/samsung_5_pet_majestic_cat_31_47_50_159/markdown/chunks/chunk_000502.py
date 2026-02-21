from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 평균공시이율+1%를 연단위\n'
 '- 복리로 계산한 금액을 더하여 지급합니다. 다만, 회사는 계약자가 제1회 보험료를 신\n'
 '- 용카드로 납입한 특별약관의 승낙을 거절하는 경우에는 신용카드의 매출을 취소하며\n'
 '- 이자를 더하여 지급하지 않습니다.\n'
 '# 제15조 (사기에 의한 계약)계약자 또는 피보험자의 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경우에는\n'
 '계약체결일부터 5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습니'),
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
