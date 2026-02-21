from langchain_core.documents import Document

chunk = Document(
    page_content=('| 3) 한 귀의 청력을 완전히 잃었을 때 | 25 |\n'
 '| 4) 한 귀의 청력에 심한 장해를 남긴 때 | 15 |\n'
 '| 5) 한 귀의 청력에 약간의 장해를 남긴 때 | 5 |\n'
 '| 6) 한 귀의 귓바퀴의 대부분이 결손된 때 | 10 |\n'
 '| 7) 평형기능에 장해를 남긴 때 | 10 |\n'
 '# 나. 장해판정기준1) 청력장해는 순음청력검사 결과에 따라 데시벨(dB:decibel)로서 표시하고, 3회\n'
 '이상 청력검사를 실시한 후 적용한다. 다만, 각 측정치의 결과값 차이가'),
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
