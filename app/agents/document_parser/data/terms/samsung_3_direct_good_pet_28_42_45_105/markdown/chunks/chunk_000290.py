from langchain_core.documents import Document

chunk = Document(
    page_content=('원금 100원, 연간 10% 이자율 적용시 연단위 복리로 계산한 2년 시점의 총 이자 금액∙ 1년차 이자 = 100원(※원금) ×10% '
 '= 10원∙ 2년차 이자 = (100원 + 10원)(※원금+1년차 이자) ×10% = 11원- 66 -66 / 181# → 2년 시점의 '
 '총 이자금액 = 10원 + 11원 = 21원- 2. 평균공시이율: 전체 보험회사 공시이율의 평균으로, 이 계약 체결 시점의 이율을\n'
 '- 말합니다. 이 평균공시이율은 금융감독원 홈페이지(www.fss.or.kr)의 「업무자료/'),
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
