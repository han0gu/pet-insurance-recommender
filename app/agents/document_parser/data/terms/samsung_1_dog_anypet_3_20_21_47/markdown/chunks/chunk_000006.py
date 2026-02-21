from langchain_core.documents import Document

chunk = Document(
    page_content=('- 함합니다)이 있을 경우 비율에 따라 손해를 보상합니다.\n'
 '- 4 -당신에게 좋은보험 삼성화재【공제계약】 공제사업을 실시하는 경영주체(협동조합 등)와 공제계약자(일반적으로 조합원) 사이\n'
 '에 체결되는 계약으로, 공제계약자들이 단체에 일정금액을 적립해두고 우연한 사고가 발생한 경우\n'
 '적립금에서 이를 구제함으로써 상호부조를 도모하는 계약을 말합니다.라. 대위권: 회사가 보험금을 지급하고 취득하는 법률상의 권리를 '
 '말합니다.# 4. 이자율 관련 용어가. 연단위 복리: 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를 원금에'),
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
