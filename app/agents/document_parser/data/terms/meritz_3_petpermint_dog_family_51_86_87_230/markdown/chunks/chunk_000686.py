from langchain_core.documents import Document

chunk = Document(
    page_content=('- 해판정 직전 1년 이상 지속적인 정신건강의학과의\n'
 '- 치료를 받았으며 GAF 30점 이하인 상태를 말한다.\n'
 '- 라) “정신행동에 심한 장해를 남긴 때”라 함은 장해\n'
 '- 판정 직전 1년 이상 지속적인 정신건강의학과의 치\n'
 '- 료를 받았으며 GAF 40점 이하인 상태를 말한다.\n'
 '- 마) “정신행동에 뚜렷한 장해를 남긴 때”라 함은 장\n'
 '- 해판정 직전 1년 이상 지속적인 정신건강의학과의\n'
 '- 치료를 받았으며, 보건복지부고시「장애정도판정기\n'
 '- 준」의“능력장애측정기준”주) 상 6개 항목 중 3개'),
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
