from langchain_core.documents import Document

chunk = Document(
    page_content=('바에 따라 가입할 수 있도록 하여 보험계약의 보험기간 중 위험에 대한 보장을 받을 수\n'
 '있는 것을 주된 내용으로 합니다.# 제 3조 (특별약관의 부가조건)① 이 특별약관에 의하여 부가하는 계약조건은 피보험자의 건강상태, '
 '위험의 종류 및 정\n'
 '도에 따라 다음 중 한가지의 방법으로 부가합니다.# 1. 할증보험료법할증위험률에 의한 보험료와 표준체 보험료와의 차액을 특별약관보험료라 '
 '하며 보\n'
 '험계약을 체결할 때의 위험의 정도에 따라 표준체 보험료에 회사에서 정한 특별약\n'
 '관보험료(보험계약이 갱신되는 경우에는 갱신시점의 표준체 보험요율을 기준으로'),
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
