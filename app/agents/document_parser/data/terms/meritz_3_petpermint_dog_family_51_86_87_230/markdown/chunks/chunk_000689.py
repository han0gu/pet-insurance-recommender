from langchain_core.documents import Document

chunk = Document(
    page_content=('- 치료를 받았으며, 보건복지부고시「장애정도판정기\n'
 '- 준」의“능력장애측정기준”상 6개 항목 중 2항목\n'
 '- 이상에서 독립적 수행이 불가능하여 타인의 도움이\n'
 '- 필요하고 GAF 70점 이하인 상태를 말한다.\n'
 '- 아) 지속적인 정신건강의학과의 치료란 3개월 이상 약\n'
 '- 물치료가 중단되지 않았음을 의미한다.\n'
 '227- 자) 심리학적 평가보고서는 정신건강의학과 의료기관에\n'
 '- 서 실시되어져야 하며, 자격을 갖춘 임상심리전문\n'
 '- 가가 시행하고 작성하여야 한다.\n'
 '- 차) 정신행동장해 진단 전문의는 정신건강의학과 전문\n'
 '- 의를 말한다.'),
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
