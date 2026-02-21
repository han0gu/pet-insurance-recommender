from langchain_core.documents import Document

chunk = Document(
    page_content=('- 로 총배기량 또는 정격출력의 크기와 관계없이 1인 또는 2인의 사람을 운송하기에 적\n'
 '- 합하게 제작된 이륜의 자동차 및 그와 유사한 구조로 되어 있는 자동차를 말하며, 도\n'
 "- 로교통법(하위 법령, 규칙 포함)에 정한 '원동기장치자전거'를 포함합니다.\n"
 '<용어풀이>개인형 이동장치(전동킥보드, 전동이륜평행차, 전동기의 동력만으로 움직일 수 있는 자전거 등을\n'
 '포함하며, 장애인 또는 교통약자가 사용하는 보행 보조용 의자차인 전동휠체어, 의료용 스쿠터 등'),
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
