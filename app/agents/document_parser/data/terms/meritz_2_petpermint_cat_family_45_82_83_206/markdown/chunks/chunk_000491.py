from langchain_core.documents import Document

chunk = Document(
    page_content=('171| 구 분 | 특정질병 | 분류코드 | 항목명 |\n'
 '| --- | --- | --- | --- |\n'
 '|  |  | GAA004 | 외이도염 (원인 불명) |\n'
 '| GAA006 | 외이염 |  |  |\n'
 '| GBA001 | 중이염 |  |  |\n'
 '| GCA001 | 내이염 |  |  |\n'
 '| LAA001 | 농피증 / 세균성 피부염 |  |  |\n'
 '| LAA002 | 말라세지아 피부염 |  |  |\n'
 '| LAA003 | 피부 사상균증 · 곰팡이성 피부염 |  |  |\n'
 '| LAA004 | 모낭염 |  |  |'),
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
