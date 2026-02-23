from langchain_core.documents import Document

chunk = Document(
    page_content=('카드매출 승인에 필요한 정보를 회사에 제공한 때가 제1회 보험료 등을 납입한 때가 되나, 계약자- 13 -당신에게 좋은보험 삼성화재의 '
 '책임있는 사유로 자동이체 또는 매출승인이 불가능한 경우에는 제1회 보험료 등이 납입되지 않\n'
 '은 것으로 봅니다.⑤ 계약이 갱신되는 경우에는 제1항 내지 제3항에 의한 보장은 기존 계약에 의한 보장이 종료하는 때\n'
 '부터 적용합니다.제22조(제2회 이후 보험료의 납입)계약자는 제2회 이후의 보험료를 납입기일까지 납입하여야 하며, 회사는 계약자가 '
 '보험료를 납입한 경'),
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
