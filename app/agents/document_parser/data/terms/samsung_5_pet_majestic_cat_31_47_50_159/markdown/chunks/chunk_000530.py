from langchain_core.documents import Document

chunk = Document(
    page_content=('실을 안 날부터 1개월 이내에 계약을 해지할 수 있습니다. 다만, 이 경우에도 회사는 실제 발생한\n'
 '보험금 지급사유에 대해서는 보험금을 지급합니다.② 회사가 제1항에 따라 이 특별약관을 해지한 경우 회사는 그 취지를 계약자에게 통\n'
 '지하고 이 특별약관의 해약환급금을 지급합니다.# 제26조 (회사의 손해배상책임)① 회사는 계약과 관련하여 임직원, 보험설계사 및 대리점의 '
 '책임있는 사유로 계약자 및\n'
 '피보험자에게 발생된 손해에 대하여 관계 법령 등에 따라 손해배상의 책임을 집니다.'),
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
