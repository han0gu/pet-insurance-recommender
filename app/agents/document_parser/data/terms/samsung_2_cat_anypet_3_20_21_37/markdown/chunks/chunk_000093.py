from langchain_core.documents import Document

chunk = Document(
    page_content=('기타지급금을 합한 금액이 1인당 "1억원까지"(본 보험회사의 여타 보호상품과 합산) 보호됩니다. 이와 별도로\n'
 '본 보험회사 보호상품의 사고보험금을 합산한 금액이 1인당 "1억원까지" 보호됩니다. 다만, 보험계약자 및 보\n'
 '험료납부자가 법인인 보험계약의 경우에는 보호되지 않습니다.- 20 -당신에게 좋은보험 삼성화재반려묘보험 애니펫특별약관- 21 -당신에게 '
 '좋은보험 삼성화재# 비뇨기질환 확장보장 특별약관# 제1 조(보상하는 손해)회사는 보통약관 제5조(보상하지 않는 손해) 제2항 제1호에도 '
 '불구하고, 비뇨기질환(요로결석 등)'),
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
