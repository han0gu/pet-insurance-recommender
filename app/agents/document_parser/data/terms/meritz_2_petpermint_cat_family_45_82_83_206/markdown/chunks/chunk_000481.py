from langchain_core.documents import Document

chunk = Document(
    page_content=('| 3 | 순환기 질환 | HAA006 | 심비대 (원인 불명) |\n'
 '| 3 | 순환기 질환 | HAA007 | 확장성 심근병증 |\n'
 '| 3 | 순환기 질환 | HAA008 | 비대성 심근병증 |\n'
 '| 3 | 순환기 질환 | HAA009 | 제한성 심근병증 |\n'
 '| 3 | 순환기 질환 | HAA010 | 일시적 심근비대증 |\n'
 '| 3 | 순환기 질환 | HAA011 | 기타 심근증 |\n'
 '| 3 | 순환기 질환 | HAA012 | 대동맥 협착증 · AS |\n'
 '| 3 | 순환기 질환 | HAA013 | 폐동맥 협착 · PS |'),
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
