from langchain_core.documents import Document

chunk = Document(
    page_content=('- 후각기능을 완전히 잃은 경우를 말하며, 후각감퇴는 장해의 대상으로 하지 않\n'
 '- 는다.\n'
 '- 3) 양쪽 코의 후각기능은 후각인지검사, 후각역치검사 등을 통해 6개월 이상 고정\n'
 '- 된 후각의 완전손실이 확인되어야 한다.\n'
 '- 4) 코의 추상(추한 모습)장해를 수반한 때에는 기능장해의 지급률과 추상장해의 지\n'
 '- 급률을 합산한다.\n'
 '# 4. 씹어먹거나 말하는 장해# 가. 장해의 분류| 장 해 의 분 류 | 지급률(%) |\n'
 '| --- | --- |\n'
 '| 1) 씹어먹는 기능과 말하는 기능 모두에 심한 장해를 남긴 때 | 100 |'),
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
