from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의사결정을 할 수 없는 상태에서 자신을 해친 경우에는 보험금의 지급사유에서 정\n'
 '- 한 해당 보험금을 지급합니다.\n'
 '- 2. 보험수익자가 고의로 피보험자를 해친 경우. 다만, 그 보험수익자가 보험금의 일부\n'
 '- 보험수익자인 경우에는 다른 보험수익자에 대한 보험금은 지급합니다.\n'
 '- 3. 계약자가 고의로 피보험자를 해친 경우\n'
 '- 4. 피보험자의 임신, 출산(제왕절개를 포함합니다), 산후기. 그러나 회사가 보장하는\n'
 '- 보험금 지급사유와 보험계약일로부터 2년이 지난 후에 발생한 습관성 유산, 불임'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
