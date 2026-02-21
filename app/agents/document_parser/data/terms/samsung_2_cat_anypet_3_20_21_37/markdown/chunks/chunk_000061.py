from langchain_core.documents import Document

chunk = Document(
    page_content=('3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을 청약할 수 있습니다. 이 경우 회사\n'
 '가 그 청약을 승낙한 때에는 계약자는 부활(효력회복)을 청약한 날까지의 연체된 보험료에 보험개\n'
 '발원이 공시하는 월평균 정기예금이율 +1% 범위내에서 각 상품별로 회사가 정하는 이율로 계산한\n'
 '금액을 더하여 납입하여야 합니다.【설명】 현재 시점의 정기예금이율은 보험개발원 홈페이지 (www.kidi.or.kr)에서 확인할 수 '
 '있습니다.- 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는 제12조(계약 전 알릴의무), 제14조(사기에'),
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
