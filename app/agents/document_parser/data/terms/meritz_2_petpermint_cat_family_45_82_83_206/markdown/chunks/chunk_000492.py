from langchain_core.documents import Document

chunk = Document(
    page_content=('| LAA004 | 모낭염 |  |  |\n'
 '| LAA005 LAA006 | 모낭충증 |  |  |\n'
 '| LAA007 | 식이 알러지 알러지 피부염 (항원 특이적) |  |  |\n'
 '| LAA008 | 아토피 (만성 피부염) |  |  |\n'
 '| LAA009 | 지루성 피부염 |  |  |\n'
 '| LAA010 | 피하 농양 |  |  |\n'
 '| LAA011 | 지방층염 |  |  |\n'
 '| LAA012 | 호산구성 육아종 |  |  |\n'
 '| LAA013 | 홍반루프스 |  |  |\n'
 '| LAA014 | 천포창 |  |  |'),
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
