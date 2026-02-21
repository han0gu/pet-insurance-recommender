from langchain_core.documents import Document

chunk = Document(
    page_content=('- 83 -7. 기타 수술의 정의에 해당하지 않는 시술| <예시안내> |\n'
 '| --- |\n'
 '| [기타 수술의 정의에 해당하지 않는 시술] - 체외 충격파 쇄석술 - 변연절제를 동반하지 않은 단순 창상봉합술 - 절개, 배농 또는 '
 '도관삽입술 - 중이내튜브유치술(중이내 환기관 삽입술) - 추간판 관련 경막외 신경차단술 - 치, 치수, 치은, 치근, 치조골의 처치 ※ '
 '본 시술들은 수술의 정의에 해당하지 않는 시술의 예시로, 예시에 기재되어 있지 않다 하더라도 수술의 정의에 해당하지 않는 경우 보상되지 '
 '않습니다. |'),
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
