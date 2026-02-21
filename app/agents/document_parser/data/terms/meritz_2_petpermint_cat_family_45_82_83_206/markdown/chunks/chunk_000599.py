from langchain_core.documents import Document

chunk = Document(
    page_content=('- 된 경우를 말하며, 다른 네 발가락에 있어서는 중족지\n'
 '- 관절의 신전운동범위만을 평가하여 정상운동범위의 1/2\n'
 '- 이하로 제한된 경우를 말한다.\n'
 '- 7) 한 발가락에 장해가 생기고 다른 발가락에 장해가 발\n'
 '- 생한 경우, 지급률은 각각 적용하여 합산한다.\n'
 '- 8) 발가락 관절의 운동범위 측정은 장해평가시점의 ｢산업\n'
 '- 재해보상보험법 시행규칙｣ 제47조 제1항 및 제3항의 정\n'
 '- 상인의 신체 각 관절에 대한 평균 운동가능영역을 기준\n'
 '- 으로 정상각도 및 측정방법 등을 따른다.\n'
 '![image](/image/placeholder)'),
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
