from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약의 갱신은 회사가 사업방법서에서 정한 갱신형 계약의 갱신종료나이 계약해\n'
 '- 당일까지로 합니다.\n'
 '- 나. 가.목에도 불구하고 갱신일부터 회사가 사업방법서에서 정한 갱신종료나이의\n'
 '- 계약해당일까지가 가.목의 보험기간 미만일 경우 그 잔여기간을 보험기간으로\n'
 '- 하여 갱신되는 것으로 하며, 세부사항은 회사의 사업방법서를 따릅니다.\n'
 '- 다. 동일한 사고에 대하여 갱신전 계약에서 이미 보험금 지급사유가 발생하여 해당\n'
 '- 보험금이 지급된 경우에는 갱신계약에서 보상하지 않습니다.'),
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
