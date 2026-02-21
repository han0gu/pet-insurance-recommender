from langchain_core.documents import Document

chunk = Document(
    page_content=('세의 경우는 6× 8cm(1/2 크기는 24cm2, 1/4 크기는 12cm2), 6세 미만의 경우는\n'
 '4× 6cm(1/2 크기는 12cm2, 1/4 크기는 6cm2)로 간주한다.# 6. 척추(등뼈)의 장해# 가. 장해의 분류| 장 해 의 '
 '분 류 | 지급률(%) |\n'
 '| --- | --- |\n'
 '| 1) 척추(등뼈)에 심한 운동장해를 남긴 때 | 40 |\n'
 '| 2) 척추(등뼈)에 뚜렷한 운동장해를 남긴 때 | 30 |\n'
 '| 3) 척추(등뼈)에 약간의 운동장해를 남긴 때 | 10 |\n'
 '| 4) 척추(등뼈)에 심한 기형을 남긴 때 | 50 |'),
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
