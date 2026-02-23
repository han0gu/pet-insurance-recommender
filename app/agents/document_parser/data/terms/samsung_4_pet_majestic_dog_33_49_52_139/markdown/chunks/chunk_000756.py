from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하고 회사의 승낙을 얻어 제1조(적용대상)의 보험수익자의 대리인으로서 보험금(사망\n'
 '- 보험금 제외)을 청구하고 수령할 수 있습니다. 다만, 2인의 청구대리인이 지정된 경우\n'
 '- 에는 그 중 대표대리인이 보험금을 청구하고 수령할 수 있으며, 대표대리인이 사망 등\n'
 '- 의 사유로 보험금 청구가 불가능한 경우에는 대표가 아닌 청구대리인도 보험금을 청\n'
 '- 구하고 수령할 수 있습니다.\n'
 '- ② 회사가 보험금을 지정대리청구인에게 지급한 경우에는 그 이후 보험금 청구를 받더라\n'
 '- 도 회사는 이를 지급하지 않습니다.\n'
 '- 제6조 (보험금의 청구)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
