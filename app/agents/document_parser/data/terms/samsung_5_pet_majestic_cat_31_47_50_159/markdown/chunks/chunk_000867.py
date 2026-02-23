from langchain_core.documents import Document

chunk = Document(
    page_content=('병의 해당 여부를 판단합니다.| 분류항목 | 분류번호 |\n'
 '| --- | --- |\n'
 '| 1 . 음식의 유해작용으로 인한 아나필락시스쇼크 | T78.0 |\n'
 '| 2 . 상세불명의 아나필락시스쇼크 | T78.2 |\n'
 '| 3 . 혈청에 의한 아나필락시스쇼크 | T80.5 |\n'
 '| 4 . 적절히 투여된 올바른 약물 또는 약제의 유해작용에 의한 아나필락시스쇼크 | T88.6 |'),
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
