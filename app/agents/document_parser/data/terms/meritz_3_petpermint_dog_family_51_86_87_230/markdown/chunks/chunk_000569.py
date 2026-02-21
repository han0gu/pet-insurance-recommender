from langchain_core.documents import Document

chunk = Document(
    page_content=('| OAA005 | 신장 결석 |  |  |\n'
 '| OAA006 | 방광염 |  |  |\n'
 '| OAA007 | 방광 결석 |  |  |\n'
 '| OAA008 | 요도 폐색 |  |  |\n'
 '| OAA009 | 요로 결석증 |  |  |\n'
 '| OAA010 | 신경성 배뇨 이상 |  |  |\n'
 '| OAA014 | 기타 비뇨기계 질환 |  |  |\n'
 '| QGA001 | 혈뇨 (원인 불명) |  |  |\n'
 '| QGA002 QGA003 | 요실금 (원인 불명) 비정상 성분의 소변 (원인 불명) |  |  |\n'
 '| QGA004 | 핍뇨 (원인 불명) |  |  |'),
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
