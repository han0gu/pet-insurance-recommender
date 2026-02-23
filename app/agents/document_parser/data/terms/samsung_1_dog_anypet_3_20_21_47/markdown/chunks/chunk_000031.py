from langchain_core.documents import Document

chunk = Document(
    page_content=('초과할 때에는 회사는 아래에 따라 손해를 보상합니다.| 다른 계약이 없을 때 이 계약의 보상책임액 손해액(피보험자가 부담한 총비용) × '
 '다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액 |\n'
 '| --- |'),
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
