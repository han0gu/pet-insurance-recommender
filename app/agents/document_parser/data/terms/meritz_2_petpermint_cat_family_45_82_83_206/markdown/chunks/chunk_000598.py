from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5) “발가락뼈 일부를 잃었을 때”라 함은 첫째 발가락\n'
 '- 에서는 지관절, 다른 네 발가락에서는 제1지관절(근\n'
 '- 위지관절)부터 심장에서 먼 쪽으로 발가락 뼈 일부가\n'
 '- 절단된 경우를 말하며, 뼈 단면이 불규칙해진 상태나\n'
 '- 발가락 길이의 단축 없이 골편만 떨어진 상태는 해당하\n'
 '- 지 않는다.\n'
 '- 6) “발가락에 뚜렷한 장해를 남긴 때”라 함은 첫째 발\n'
 '- 가락의 경우에 중족지관절과 지관절의 굴신(굽히고 펴\n'
 '- 기)운동범위 합계가 정상 운동가능영역의 1/2 이하가\n'
 '- 된 경우를 말하며, 다른 네 발가락에 있어서는 중족지'),
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
