from langchain_core.documents import Document

chunk = Document(
    page_content=('해당월에 보험금 지급사유 발생일이 없는 경우에는 해당월의 마지막 날로 합니다)에 반려\n'
 '동물 양육자금Ⅰ으로 보험수익자에게 지급합니다.# 제 2조 (보험금 지급에 관한 세부규정)① 제1조(보험금의 지급사유)의‘사망’에는 '
 '보험기간에 다음 어느 하나의 사유가 발생\n'
 '한 경우를 포함합니다.1. 실종선고를 받은 경우: 법원에서 인정한 실종기간이 끝나는 때에 사망한 것으로 봅\n'
 '니다.\n'
 '2. 관공서에서 수해, 화재나 그 밖의 재난을 조사하고 사망한 것으로 통보하는 경우:\n'
 '가족관계등록부에 기재된 사망연월일을 기준으로 합니다.- \n'
 '<용어풀이>-'),
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
