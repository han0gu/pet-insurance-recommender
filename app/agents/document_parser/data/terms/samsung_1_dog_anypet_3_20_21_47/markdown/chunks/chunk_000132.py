from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 회사는 제1항에 열거하는 서류 이외의 서류 제출을 요구할 수 있습니다.\n'
 '제4조(준용규정)이 특별약관에서 정하지 않은 사항은 보통약관을 따릅니다.30 -당신에게 좋은보험 삼성화재# 치료비보상 제외 특별약관제1 '
 '조(보험금을 지급하지 않는 사유)회사는 보통약관 제4조(보상하는 손해)에도 불구하고 상해 또는 질병 치료비로 인한 보험금은 이 특별'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
