from langchain_core.documents import Document

chunk = Document(
    page_content=('- 익자의 책임있는 사유로 보험금 지급사유의 조사 및 확인이 지연되는 경우\n'
 '- 33 -6. 제4조(보험금 지급에 관한 세부규정) 제4항에 따라 보험금 지급사유에 대해 제3자\n'
 '의 의견에 따르기로 한 경우<유의사항># 분쟁조정은 이 약관의 (분쟁의 조정) 조항에 따라 금융감독원에 신청할 수 있습니다.- ③ '
 '제2항에 의하여 장해지급률의 판정 및 지급할 보험금의 결정과 관련하여 확정된 장해\n'
 '- 지급률에 따른 보험금을 초과한 부분에 대한 분쟁으로 보험금 지급이 늦어지는 경우'),
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
