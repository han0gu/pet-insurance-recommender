from langchain_core.documents import Document

chunk = Document(
    page_content=('체납된 세금에 대하여 가산금 징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니다.\n'
 '국세 및 지방세 체납시 국세청 및 지방자치단체에 의해 채무자의 해약환급금이 압류될 수 있으며,\n'
 '체납처분 절차에 따라 회사는 채권자에게 해약환급금을 지급하게 됩니다.- ② 회사는 제1항에 따른 계약자 명의변경 신청 및 특별약관의 '
 '특별부활(효력회복) 청약을\n'
 '- 승낙합니다.\n'
 '- ③ 회사는 제1항의 통지를 지정된 보험수익자에게 하여야 합니다. 다만, 회사는 법정상속'),
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
