from langchain_core.documents import Document

chunk = Document(
    page_content=('원인으로 보험계약의 보험금 지급사유가 발생하였을 경우에는 보험계약의 규정에\n'
 '도 불구하고 계약을 체결할 때 정한 삭감기간에 따라 다음과 같이 보험금을 지급\n'
 '합니다.| 경과기간 | 기준 | 삭감기간별 보험금지급비율 | 삭감기간별 보험금지급비율 | 삭감기간별 보험금지급비율 | 삭감기간별 '
 '보험금지급비율 | 삭감기간별 보험금지급비율 |\n'
 '| --- | --- | --- | --- | --- | --- | --- |\n'
 '| 경과기간 | 기준 | 1년 | 2년 | 3년 | 4년 | 5년 |'),
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
