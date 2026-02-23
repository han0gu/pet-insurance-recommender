from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5. 제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것\n'
 '제1항에 따라 계약이 해지된 경우에는 제30조(보험료의 환급)에 따라 보험료를 계약자에게 지급합\n'
 '니다.- 14 -당신에게 좋은보험 삼성화재# 제24조[보험료의 납입연체로 인한 해지계약의 부활(효력회복)]① 제23조[보험료의 납입이 '
 '연체되는 경우 납입최고(독촉)와 계약의 해지]에 따라 계약이 해지되었으\n'
 '나 계약자가 제30조(보험료의 환급)에 따라 보험료를 돌려받지 않는 경우 계약자는 해지된 날부터'),
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
