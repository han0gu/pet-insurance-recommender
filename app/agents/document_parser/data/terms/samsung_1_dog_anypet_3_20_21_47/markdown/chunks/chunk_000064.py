from langchain_core.documents import Document

chunk = Document(
    page_content=('금액을 더하여 납입하여야 합니다.【설명】 현재 시점의 정기예금이율은 보험개발원 홈페이지 (www.kidi.or.kr)에서 확인할 수 '
 '있습니다.- 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는 제12조(계약 전 알릴의무), 제14조(사기에\n'
 '- 의한 계약), 제15조(보험계약의 성립), 제21조(제1회 보험료 등 및 회사의 보장개시) 및 제26조(계\n'
 '- 약의 해지)의 규정을 준용합니다. 이 때 회사는 해지 전 발생한 보험금 지급사유를 이유로 부활(효\n'
 '- 력회복)을 거절하지 않습니다.'),
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
