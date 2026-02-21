from langchain_core.documents import Document

chunk = Document(
    page_content=('| GAA003 | 외이도염 (알러지성) |  |  |\n'
 '| GAA004 | 외이도염 (원인 불명) |  |  |\n'
 '| GAA006 | 외이염 |  |  |\n'
 '| GBA001 | 중이염 |  |  |\n'
 '| GCA001 | 내이염 |  |  |\n'
 '| LAA001 | 농피증 / 세균성 피부염 |  |  |\n'
 '| LAA002 | 말라세지아 피부염 |  |  |\n'
 '| LAA003 | 피부 사상균증 · 곰팡이성 피부염 |  |  |\n'
 '| LAA004 | 모낭염 |  |  |\n'
 '| LAA005 | 모낭충증 |  |  |'),
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
