from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5) 순음청력검사를 실시하기 곤란하거나(청력의 감소가 의심되지만 의사소\n'
 '- 통이 되지 않는 경우, 만 3세 미만의 소아 포함) 검사결과에 대한 검증\n'
 '- 이 필요한 경우에는 ‘언어청력검사, 임피던스 청력검사, 청성뇌간반응\n'
 '- 141 -KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 141별표사항검사(ABR), 이음향방사검사’ 등을 추가실시 후 장해를 '
 '평가한다.- 다. 귓바퀴의 결손\n'
 '- 1) ‘귓바퀴의 대부분이 결손된 때’라 함은 귓바퀴의 연골부가 1/2 이상 결\n'
 '- 손된 경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
