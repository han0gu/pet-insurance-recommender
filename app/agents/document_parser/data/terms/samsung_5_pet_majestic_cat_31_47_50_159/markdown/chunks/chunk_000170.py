from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 수사기관의 조사\n'
 '- 4. 해외에서 발생한 보험사고에 대한 조사\n'
 '- 5. 제6항에 따른 회사의 조사요청에 대한 동의 거부 등 계약자, 피보험자 또는 보험수\n'
 '- 익자의 책임있는 사유로 보험금 지급사유의 조사 및 확인이 지연되는 경우\n'
 '- 6. 각 특별약관별 보험금 지급에 관한 세부규정에 따라 보험금 지급사유에 대해 제3\n'
 '- 자의 의견에 따르기로 한 경우\n'
 '- 7. 제6조(보험료 납입면제에 관한 세부규정)에 따라 보험료 납입면제 사유에 대해 제3\n'
 '- 자의 의견에 따르기로 한 경우'),
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
