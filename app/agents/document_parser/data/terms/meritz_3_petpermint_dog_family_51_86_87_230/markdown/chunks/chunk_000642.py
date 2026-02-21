from langchain_core.documents import Document

chunk = Document(
    page_content=('215| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 를 남긴 때 6) 한팔의 3대관절중 관절 하나의 기능에 약간의 장해 를 남긴 때 7) 한팔에 가관절이 남아 뚜렷한 장해를 남긴 때 '
 '8) 한팔에 가관절이 남아 약간의 장해를 남긴 때 9) 한팔의 뼈에 기형을 남긴 때 | 5 20 10 5 |\n'
 '# 나. 장해판정기준- 1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것\n'
 '- 이 기능장해의 원인이 되는 때에는 그 내고정물 등이\n'
 '- 제거된 후 장해를 평가한다. 단, 제거가 불가능한 경'),
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
