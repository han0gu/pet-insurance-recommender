from langchain_core.documents import Document

chunk = Document(
    page_content=('- (눈, 귀, 코, 팔, 다리 등)는 해당 장해로도 평가\n'
 '- 하고 그 중 높은 지급률을 적용한다.\n'
 '- 라) 뇌졸중, 뇌손상, 척수 및 신경계의 질환 등은 발\n'
 '- 병 또는 외상 후 12개월 동안 지속적으로 치료한\n'
 '- 후에 장해를 평가한다.\n'
 '- 그러나, 12개월이 지났다고 하더라도 뚜렷하게\n'
 '- 기능 향상이 진행되고 있는 경우 또는 단기간내\n'
 '- 에 사망이 예상되는 경우는 6개월의 범위에서 장\n'
 '- 해 평가를 유보한다.\n'
 '- 마) 장해진단 전문의는 재활의학과, 신경외과 또는\n'
 '- 신경과 전문의로 한다.'),
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
