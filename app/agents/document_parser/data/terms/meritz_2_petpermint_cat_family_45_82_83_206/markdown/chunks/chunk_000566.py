from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계없이 전체를 일괄하여 하나의 장해로 취급한다. 다발\n'
 '- 성늑골 기형의 경우 각각의 각(角) 변형을 합산하지 않\n'
 '- 고 그 중 가장 높은 각(角) 변형을 기준으로 평가한다.\n'
 '189| ![image](/image/placeholder)\n'
 ' |\n'
 '| --- |\n'
 '| < 가슴뼈 > |\n'
 '| ![image](/image/placeholder)\n'
 ' |\n'
 '| < 골반뼈 > |\n'
 '# 8. 팔의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 두팔의 손목이상을 잃었을 때 | 100 |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
