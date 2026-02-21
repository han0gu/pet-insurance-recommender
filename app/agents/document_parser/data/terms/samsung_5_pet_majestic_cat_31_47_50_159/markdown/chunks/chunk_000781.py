from langchain_core.documents import Document

chunk = Document(
    page_content=('# 7. 체간골의 장해- \n'
 '# 가. 장해의 분류장 해 의 분 류 지급률(%)\n'
 '1) 어깨뼈(견갑골)나 골반뼈(장골, 제2천추 이하의 천골, 미골, 좌골 포함)에 뚜 15\n'
 '렷한 기형을 남긴 때\n'
 '2) 빗장뼈(쇄골), 가슴뼈(흉골), 갈비뼈(늑골)에 뚜렷한 기형을 남긴 때 10# 나. 장해판정기준1) "체간골" 이라 함은 '
 '어깨뼈(견갑골), 골반뼈(장골, 제2천추 이하의 천골, 미골,- 141 -# 좌골 포함), 빗장뼈(쇄골), 가슴뼈(흉골), '
 '갈비뼈(늑골)를 말하며 이를 모두 동'),
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
