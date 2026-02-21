from langchain_core.documents import Document

chunk = Document(
    page_content=('- 않은 기타 장해상태에 대해서는 “<붙임>일상생활\n'
 '- 기본동작(ADLs) 제한 장해평가표”에 해당하는 장해\n'
 '- 가 있을 때 ADLs 장해 지급률을 준용한다.\n'
 '- 8) 상기 장해항목에 해당되지 않는 장기간의 간병이 필요\n'
 '- 한 만성질환(만성간질환, 만성폐쇄성폐질환 등)은 장\n'
 '- 해의 평가 대상으로 인정하지 않는다.\n'
 '13. 신경계ㆍ정신행동 장해가. 장해의 분류225| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 신경계에 장해가 남아 일상생활 기본동작에 제한을 남긴 때 | 10∼100 |'),
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
