from langchain_core.documents import Document

chunk = Document(
    page_content=('- 있을 정도로 방사선 검사로 측정한 각(角) 변형\n'
 '- 이 20° 이상인 경우\n'
 '- 다) 미골의 기형은 골절이나 탈구로 방사선 검사로 측\n'
 '- 정한 각(角) 변형이 70° 이상 남은 상태\n'
 '- 3) “빗장뼈(쇄골), 가슴뼈(흉골), 갈비뼈(늑골), 어깨\n'
 '- 뼈(견갑골)에 뚜렷한 기형이 남은 때”라 함은 방사\n'
 '- 선 검사로 측정한 각(角) 변형이 20° 이상인 경우를\n'
 '- 말한다.\n'
 '- 4) 갈비뼈(늑골)의 기형은 그 개수와 정도, 부위 등에 관\n'
 '- 계없이 전체를 일괄하여 하나의 장해로 취급한다. 다발'),
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
