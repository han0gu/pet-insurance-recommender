from langchain_core.documents import Document

chunk = Document(
    page_content=('있는 틀니 등)의 파손은 후유장해의 대상이 되지 않\n'
 '는다.# 5. 외모의 추상(추한 모습)장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 외모에 뚜렷한 추상(추한 모습)을 남긴 때 | 15 |\n'
 '| 2) 외모에 약간의 추상(추한 모습)을 남긴 때 | 5 |\n'
 '# 나. 장해판정기준- 1) “외모”란 얼굴(눈, 코, 귀, 입 포함), 머리, 목을\n'
 '- 말한다.\n'
 '- 2) “추상(추한 모습)장해”라 함은 성형수술(반흔(흉\n'
 '- 터)성형술, 레이저치료 등 포함)을 시행한 후에도'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
