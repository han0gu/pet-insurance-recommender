from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 그<br>보험수익자가 보험금의 일부 보험수익자인 경우에는<br>다른 보험수익자에 대한 보험금은 지급합니다.<br>③ 계약자가 '
 '고의로 피보험자를 해친 경우<br>④ 피보험자의 임신, 출산(제왕절개를 포함합니다), 산후<br>기'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
