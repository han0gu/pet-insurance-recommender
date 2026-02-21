from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 피보험자 의사표시의 확인방법 포함)\n'
 '② 회사는 제1항에 열거하는 서류 이외의 서류 제출을 요구할 수 있습니다.# 제4조(준용규정)이 특별약관에서 정하지 않은 사항은 '
 '보통약관을 따릅니다.- 23 -당신에게 좋은보험 삼성화재# 보험료분납 특별약관# 제1조(보험료의 납입)- ① 이 특별약관에 따라 계약자는 '
 '보험기간이 1년인 보험 계약에 대하여 보험료를 제2항에 정한 바에\n'
 '- 따라 나누어 납입할 수 있습니다.\n'
 '- ② 계약자는 이 보험의 보험료 및 해약환급금 산출방법서에서 정한 방법에 의하여 계산된 분납보험료'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
