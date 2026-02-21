from langchain_core.documents import Document

chunk = Document(
    page_content=('적용되지 않습니다.# 제4조(전환 취소)계약자는 전환대상계약에 대하여 장애인전용보험으로의 전환을 취소할 수 있으며, 이 경우 전환취소\n'
 '신청서를 회사에 제출하여야 합니다.# 제5조(준용규정)- ① 이 특약에서 정하지 않은 사항에 대하여는 전환대상계약 약관, 소득세법 등 '
 '관련법규에서 정하는\n'
 '- 바에 따릅니다.\n'
 '- ② 소득세법 등 관련법규가 제·개정 또는 폐지되는 경우 변경된 법령을 따릅니다.\n'
 '- 37 -당신에게 좋은보험 삼성화재'),
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
