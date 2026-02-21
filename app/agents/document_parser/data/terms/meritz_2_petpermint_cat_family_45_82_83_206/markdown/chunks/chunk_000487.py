from langchain_core.documents import Document

chunk = Document(
    page_content=('| QGA001 | 피부질환 | 혈뇨 (원인 불명) |  |\n'
 '| QGA002 | 피부질환 | 요실금 (원인 불명) |  |\n'
 '| QGA003 QGA004 | 피부질환 | 비정상 성분의 소변 (원인 불명) 핍뇨 (원인 불명) |  |\n'
 '| 5 |  | AFA001 | 지방종 |\n'
 '| 5 | AFA002 | 조직구종 (피부) |  |\n'
 '| 5 | AFA003 | 유두종 (피부) |  |\n'
 '| 5 | AFA004 | 피지종 |  |\n'
 '| 5 | AFA005 | 모낭상피종 |  |'),
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
