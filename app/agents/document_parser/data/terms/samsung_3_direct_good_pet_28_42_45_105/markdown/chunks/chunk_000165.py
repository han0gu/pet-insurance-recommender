from langchain_core.documents import Document

chunk = Document(
    page_content=('- 또는 잔여 보험료 납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초과할 수 없음)\n'
 '- 동안의 분납 중 선택하여 정산금액을 납입하여야 합니다. 다만, 보험료 갱신형 계약\n'
 '- 등 회사가 정하는 기준에 따라 일부 보험계약의 경우 분납이 제한될 수 있습니다.\n'
 '- ④ 제1항의 통지에 따라 위험의 증가로 보험료를 더 내야 할 경우 회사가 청구한 추가보\n'
 '- 험료(정산금액을 포함합니다)를 계약자가 납입하지 않았을 때, 회사는 위험이 증가되\n'
 '- 기 전에 적용된 보험요율(이하「변경전 요율」이라 합니다)의 위험이 증가된 후에 적'),
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
