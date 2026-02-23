from langchain_core.documents import Document

chunk = Document(
    page_content=('의원 및 조산원으로 나누어집니다.# 제5조(보험금의 분담)① 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 '
 '포함합\n'
 '니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 보상\n'
 '책임액의 합계액이 손해액을 초과할 때에는 회사는 아래에 따라 손해를 보상합니다.# <용어풀이># [공제계약]유사보험으로서 공제 사업을 '
 '실시하는 경영주체와 공제 계약자 사이에 체결되는 계약을 말합니다.\n'
 '우체국, 신협, 새마을금고 등이 공제계약을 취급합니다.이 계약의 보상책임액\n'
 '손해액 ×'),
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
