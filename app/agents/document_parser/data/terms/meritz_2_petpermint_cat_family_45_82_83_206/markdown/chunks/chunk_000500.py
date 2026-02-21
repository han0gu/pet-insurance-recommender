from langchain_core.documents import Document

chunk = Document(
    page_content=('| KGA005 | 촌충증 |  |  |\n'
 '| KGA006 | 간충증 |  |  |\n'
 '| KGA007 | 기타 소화기계 기생충증 |  |  |\n'
 '| KGA008 | 기타 소화기계 감염증 |  |  |\n'
 '| KGA009 | 소화계통의 기타 질환 |  |  |\n'
 '| PAA014 | 고양이 파보 바이러스(FPV) 고양이 코로나 바이러스 감염 |  |  |\n'
 '| PAA015 QEA001 | 구토 (원인 불명) |  |  |\n'
 '| QEA002 | 설사 / 혈변 (원인 불명) |  |  |\n'
 '| QEA003 | 복통 (원인 불명) |  |  |'),
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
