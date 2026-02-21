from langchain_core.documents import Document

chunk = Document(
    page_content=('2) 평형기능의 장해는 장해판정 직전 1년 이상 지속적인 치료 후 장해가 고착되었\n'
 '을 때 판정하며, 뇌병변 여부, 전정기능 이상 및 장해상태를 평가하기 위해 아\n'
 '래의 검사들을 기초로 한다.- 가) 뇌영상검사(CT, MRI)\n'
 '- 나) 온도안진검사, 전기안진검사(또는 비디오안진검사) 등\n'
 '# 3. 코의 장해가. 장해의 분류| 장 해 의 분 류 | 지급률(%) |\n'
 '| --- | --- |\n'
 '| 1) 코의 호흡기능을 완전히 잃었을 때 | 15 |\n'
 '| 2) 코의 후각기능을 완전히 잃었을 때 | 5 |'),
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
