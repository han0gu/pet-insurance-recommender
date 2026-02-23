from langchain_core.documents import Document

chunk = Document(
    page_content=('부터 120일 이내의 치료비는 보상하여 드립니다.【파보바이러스 감염증】 파보바이러스에 감염되어 구토와 설사 등의 증상을 일으킴\n'
 '【디스템퍼바이러스 감염증】 디스템퍼바이러스에 감염되어 호흡기 질환과 신경증상을 일으킴\n'
 '【코로나바이러스 감염증】 코로나바이러스성 장염으로 불리며, 소화계통의 바이러스 감염으로 인해 구토,\n'
 '설사 등의 증상을 일으킴# 제2조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 47 -당신에게 좋은보험 삼성화재'),
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
