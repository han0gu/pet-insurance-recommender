from langchain_core.documents import Document

chunk = Document(
    page_content=('니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 보상\n'
 '책임액의 합계액이 손해액을 초과할 때에는 회사는 아래에 따라 손해를 보상합니다.| <지급보험금 계산방법> 다른 계약이 없을 때 이 계약의 '
 '지급보험금 피보험자가 부담한 의료비 × 다른 계약이 없는 것으로 하여 각각 계약의 지급보험금의 합계액 |\n'
 '| --- |\n'
 '| <용어풀이> [공제계약] 유사보험으로서 공제 사업을 실시하는 경영주체와 공제 계약자 사이에 체결되는 계약을 말합니다. 우체국, 신협, '
 '새마을금고 등이 공제계약을 취급합니다. |'),
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
