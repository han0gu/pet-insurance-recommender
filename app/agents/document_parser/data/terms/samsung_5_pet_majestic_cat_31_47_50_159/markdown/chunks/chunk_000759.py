from langchain_core.documents import Document

chunk = Document(
    page_content=('- 8) "말하는 기능에 뚜렷한 장해를 남긴 때" 라 함은 아래의 경우 중 하나 이상에\n'
 '- 해당되는 때를 말한다.\n'
 '- 가) 언어평가상 자음정확도가 50% 미만인 경우\n'
 '- 나) 언어평가상 표현언어지수 25 미만인 경우\n'
 '9) "말하는 기능에 약간의 장해를 남긴 때" 라 함은 아래의 경우 중 하나 이상에\n'
 '해당되는 때를 말한다.\n'
 '가) 언어평가상 자음정확도가 75% 미만인 경우\n'
 '나) 언어평가상 표현언어지수 65 미만인 경우- 10) 말하는 기능의 장해는 1년 이상 지속적인 언어치료를 시행한 후 증상이 고착'),
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
