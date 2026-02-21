from langchain_core.documents import Document

chunk = Document(
    page_content=('- - 환급금에 관한 사항\n'
 '- - 고지의무 및 통지의무 위반의 효과\n'
 '- - 만기시 자동갱신되는 보험계약의 경우 자동갱신의 조건\n'
 '- - 그 밖에 약관에 기재된 보험계약의 중요사항\n'
 '② 제1항과 관련하여 통신판매계약의 경우, 회사는 계약자가 가입한 특약만 포함한 약관을 드리며, 전\n'
 '화를 이용하여 체결하는 계약은 계약자의 동의를 얻어 다음의 방법으로 약관의 중요한 내용을 설\n'
 '명할 수 있습니다.1. 전화를 이용하여 청약내용, 보험료납입, 보험기간, 계약 전 알릴 의무, 약관의 중요한 내용 등'),
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
