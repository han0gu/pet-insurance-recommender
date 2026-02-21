from langchain_core.documents import Document

chunk = Document(
    page_content=('부를 판단합니다.| 분류항목 | 분류번호 |\n'
 '| --- | --- |\n'
 '| 1 . 두개골 및 안면골의 골절 | S02 |\n'
 '| (치아의 파절 및 파절치 제외) | (S02.5 제외) |\n'
 '| 2 . 머리의 으깸손상 | S07 |\n'
 '| 3 . 머리의 상세불명 손상 | S09.9 |\n'
 '| 4 . 목의 골절 | S12 |\n'
 '| 5 . 늑골, 흉골 및 흉추의 골절 | S22 |\n'
 '| 6 . 요추 및 골반의 골절 | S32 |\n'
 '| 7 . 어깨 및 위팔의 골절 | S42 |\n'
 '| 8 . 아래팔의 골절 | S52 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
