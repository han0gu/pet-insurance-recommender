from langchain_core.documents import Document

chunk = Document(
    page_content=("하지 않았을 때에는 지급기일의 다음날부터 지급일까지의 기간에 대하여 <부표> '보험금을 지급할\n"
 "때의 적립이율'에 따라 연단위 복리로 계산한 금액을 보험금에 더하여 지급합니다. 그러나 피보험자\n"
 '의 책임 있는 사유로 지급이 지연될 때에는 그 해당기간에 대한 이자를 더하여 드리지 않습니다.<부표> 보험금을 지급할 때의 적립이율 '
 '(제8조 제2항 관련)| 기 간 | 지 급 이 자 |\n'
 '| --- | --- |\n'
 '| 지급기일의 다음 날부터 30일 이내 기간 | 보험계약대출이율 |'),
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
