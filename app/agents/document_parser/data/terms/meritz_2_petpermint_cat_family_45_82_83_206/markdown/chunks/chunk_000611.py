from langchain_core.documents import Document

chunk = Document(
    page_content=('- 마) 장해진단 전문의는 재활의학과, 신경외과 또는\n'
 '- 신경과 전문의로 한다.\n'
 '# 2) 정신행동가) 정신행동장해는 보험기간중에 발생한 뇌의 질병201또는 상해를 입은 후 18개월이 지난 후에 판정\n'
 '함을 원칙으로 한다. 단, 질병발생 또는 상해를\n'
 '입은 후 의식상실이 1개월 이상 지속된 경우에\n'
 '는 질병발생 또는 상해를 입은 후 12개월이 지난\n'
 '후에 판정할 수 있다.- 나) 정신행동장해는 장해판정 직전 1년 이상 충분한 정\n'
 '- 신건강의학과의 전문적 치료를 받은 후 치료에도\n'
 '- 불구하고 장해가 고착되었을 때 판정하여야 하며,'),
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
