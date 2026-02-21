from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다) 중에 상해를 입고 그 직접적인 결과로써 [별표-상해관련3]5대골절 분류표에\n'
 '정한 골절(이하「5대골절」이라 합니다)로 진단확정된 경우에는 보험증권에 기재된\n'
 '이 특별약관의 보험가입금액을 5대골절 진단비로 보험수익자에게 지급합니다.\n'
 '② 제1항의 5대골절 진단비는 매사고마다 지급합니다. 다만, 동일한 상해사고를 직접적인\n'
 '원인으로 2가지 이상의 5대골절 상태가 발생한 경우에는 1회에 한하여 보상합니다.- \n'
 '제 2조 (보험금 지급에 관한 세부규정)보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못'),
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
