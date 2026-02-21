from langchain_core.documents import Document

chunk = Document(
    page_content=('할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합\n'
 '니다.# ③ 지급금과 이자율 관련 용어1. 연단위 복리: 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를\n'
 '원금에 더한 금액을 다음 1년의 원금으로 하는 이자 계산방법을 말합니다.<예시안내># [연단위 복리]원금 100원, 연간 10% 이자율 '
 '적용시 연단위 복리로 계산한 2년 시점의 총 이자 금액- · 1년차 이자 = 100원(※원금) ×10% = 10원'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
