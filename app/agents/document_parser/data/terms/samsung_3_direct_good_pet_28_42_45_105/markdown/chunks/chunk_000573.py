from langchain_core.documents import Document

chunk = Document(
    page_content=('| 28 | 갑상선 |\n'
 '| 29 | 부갑상선 |\n'
 '| 30 | 서혜부(넓적다리 부위의 위쪽 주변)(서혜 탈장, 음낭 탈장 또는 대퇴 탈장이 생긴 경우 에 한함) |\n'
 '| 31 | 피부(두피 및 입술 포함) |\n'
 '| 32 | 경추부(해당신경 포함) |\n'
 '| 33 | 흉추부(해당신경 포함) |\n'
 '| 34 | 요추부(해당신경 포함) |\n'
 '| 35 | 천골(엉치뼈)부 및 미골(꼬리뼈)부(해당 신경 포함) |\n'
 '| 36 | 왼쪽 어깨 |\n'
 '| 37 | 오른쪽 어깨 |\n'
 '| 38 | 왼팔(왼쪽 어깨 제외, 왼손 포함) |'),
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
