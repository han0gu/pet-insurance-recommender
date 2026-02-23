from langchain_core.documents import Document

chunk = Document(
    page_content=('- 니다.\n'
 '- 4. 한국표준질병·사인분류 : 제9차 개정 한국표준질병·사인분류(통계청 고시 제2025-\n'
 '# ③ 지급금과 이자율 관련 용어1. 연단위 복리: 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를\n'
 '원금에 더한 금액을 다음 1년의 원금으로 하는 이자 계산방법을 말합니다.# <예시안내># [연단위 복리]원금 100원, 연간 10% '
 '이자율 적용시 연단위 복리로 계산한 2년 시점의 총 이자 금액1년차 이자 = 100원(※원금) ×10% = 10원'),
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
