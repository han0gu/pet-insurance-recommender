from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4. 기타 보험수익자가 보험금의 수령 또는 보험료 납입면제 청구에 필요하여 제출하\n'
 '- 는 서류(단, 단체취급 특별약관을 부가하는 경우, 사망보험금을 지급할 때 피보험\n'
 '- 자의 법정상속인이 아닌 자가 청구하는 경우 법정상속인의 확인서 등)\n'
 '② 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나 의'),
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
