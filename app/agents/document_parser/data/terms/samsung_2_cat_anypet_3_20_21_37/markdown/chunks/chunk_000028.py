from langchain_core.documents import Document

chunk = Document(
    page_content=('나. 적용 지급한도액: 입원 1일당 지급한도액 × 입원일수 + 수술 1회당 지급한도액 × 수술횟수③ 통원 또는 통원 당일 수술이 이루어진 '
 '경우의 적용 지급한도액은 아래와 같습니다.# 1. 통원만의 경우가. 적용 자기부담금 : 1일당 자기부담금- 8 -당신에게 좋은보험 '
 '삼성화재나. 적용 지급한도액 : 통원 1일당 지급한도액# 2. 통원 당일 수술이 이루어진 경우가. 적용 자기부담금: 통원 1일당 '
 '자기부담금 + 수술 1회당 자기부담금 × 수술횟수'),
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
