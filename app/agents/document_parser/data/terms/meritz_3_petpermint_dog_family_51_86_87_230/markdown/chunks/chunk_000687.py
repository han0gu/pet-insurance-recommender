from langchain_core.documents import Document

chunk = Document(
    page_content=('- 준」의“능력장애측정기준”주) 상 6개 항목 중 3개\n'
 '- 항목 이상에서 독립적 수행이 불가능하여 타인의 도\n'
 '- 움이 필요하고 GAF 50점 이하인 상태를 말한다.\n'
 '※ 주) 능력장애측정기준의 항목 : ㉮ 적절한 음식\n'
 '섭취, ㉯ 대소변관리, 세면, 목욕, 청소 등의 청\n'
 '결 유지, ㉰ 적절한 대화기술 및 협조적인 대인\n'
 '관계, ㉱ 규칙적인 통원․약물 복용, ㉲ 소지품 및\n'
 '금전관리나 적절한 구매행위, ㉳ 대중교통이나\n'
 '일반공공시설의 이용- 바) “정신행동에 약간의 장해를 남긴 때”라 함은 장'),
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
