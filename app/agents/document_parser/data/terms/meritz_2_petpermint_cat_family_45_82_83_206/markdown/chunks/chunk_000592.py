from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 제1지관절(근위지관절)부터(제1지관절 포함) 심장\n'
 '- 에서 가까운 쪽으로 손가락이 절단되었을 때를 말한\n'
 '- 다.\n'
 '- 6) “손가락뼈 일부를 잃었을 때”라 함은 첫째 손가락\n'
 '- 의 지관절, 다른 네 손가락의 제1지관절(근위지관\n'
 '- 절)부터 심장에서 먼쪽으로 손가락 뼈의 일부가 절\n'
 '- 단된 경우를 말하며, 뼈 단면이 불규칙해진 상태나\n'
 '- 손가락 길이의 단축 없이 골편만 떨어진 상태는 해당\n'
 '- 하지 않는다.\n'
 '- 7) “손가락에 뚜렷한 장해를 남긴 때”라 함은 첫째 손가\n'
 '- 락의 경우 중수지관절 또는 지관절의 굴신(굽히고 펴'),
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
