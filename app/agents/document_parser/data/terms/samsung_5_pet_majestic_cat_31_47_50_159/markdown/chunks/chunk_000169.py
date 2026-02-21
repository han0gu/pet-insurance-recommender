from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기간이 제1항의 지급기일을 초과할 것이 명백히 예상되는 경우에는 그 구체적 사유와\n'
 '- 지급예정일 및 보험금 가지급 제도(회사가 추정하는 보험금의 50% 이내를 지급)에 대\n'
 '- 하여 계약자, 피보험자 또는 보험수익자에게 즉시 통지합니다. 다만, 지급예정일은 다\n'
 '- 음 각 호의 어느 하나에 해당하는 경우를 제외하고는 제9조(보험금 등의 청구)에서 정\n'
 '- 한 서류를 접수한 날부터 30영업일 이내에서 정합니다.\n'
 '- 1. 소송제기\n'
 '- 2. 분쟁조정 신청\n'
 '- 3. 수사기관의 조사\n'
 '- 4. 해외에서 발생한 보험사고에 대한 조사'),
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
