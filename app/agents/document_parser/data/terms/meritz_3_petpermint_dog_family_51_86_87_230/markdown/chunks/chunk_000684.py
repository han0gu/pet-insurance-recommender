from langchain_core.documents import Document

chunk = Document(
    page_content=('- 병 또는 외상 후 12개월 동안 지속적으로 치료한\n'
 '- 후에 장해를 평가한다.\n'
 '- 그러나, 12개월이 지났다고 하더라도 뚜렷하게\n'
 '- 기능 향상이 진행되고 있는 경우 또는 단기간내\n'
 '- 에 사망이 예상되는 경우는 6개월의 범위에서 장\n'
 '- 해 평가를 유보한다.\n'
 '- 마) 장해진단 전문의는 재활의학과, 신경외과 또는\n'
 '- 신경과 전문의로 한다.\n'
 '# 2) 정신행동가) 정신행동장해는 보험기간중에 발생한 뇌의 질병226또는 상해를 입은 후 18개월이 지난 후에 판정\n'
 '함을 원칙으로 한다. 단, 질병발생 또는 상해를'),
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
