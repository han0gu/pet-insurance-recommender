from langchain_core.documents import Document

chunk = Document(
    page_content=('- 흡곤란으로 지속적인 산소치료가 필요하며, 폐기\n'
 '- 능 검사(PFT)상 폐환기 기능(1초간 노력성 호기\n'
 '- 량, FEV1)이 정상예측치의 40% 이하로 저하된 때\n'
 '- 6) 흉복부, 비뇨생식기계 장해는 질병 또는 외상의 직접\n'
 '- 결과로 인한 장해를 말하며, 노화에 의한 기능장해\n'
 '- 또는 질병이나 외상이 없는 상태에서 예방적으로 장\n'
 '- 기를 절제, 적출한 경우는 장해로 보지 않는다.\n'
 '- 7) 상기 흉복부 및 비뇨생식기계 장해항목에 명기되지\n'
 '- 않은 기타 장해상태에 대해서는 “<붙임>일상생활'),
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
