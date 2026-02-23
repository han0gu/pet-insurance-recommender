from langchain_core.documents import Document

chunk = Document(
    page_content=('에는 예외로 합니다.# 제2조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 24 -당신에게 좋은보험 삼성화재# '
 '보험료 자동이체 특별약관# 제1 조(보험료납입)계약자는 제2회 이후의 보험료부터 이 특별약관에 따라 계약자의 지정계좌를 이용하여 보험료를 '
 '자동\n'
 '납입 합니다.# 제2조(보험료의 영수)자동납입일자는 이 청약서에 기재된 보험료납입 해당일에도 불구하고 회사와 계약자가 별도로 약정한'),
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
