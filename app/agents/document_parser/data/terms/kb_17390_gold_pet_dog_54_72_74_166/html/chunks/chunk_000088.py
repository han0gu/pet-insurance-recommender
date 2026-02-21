from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 진단계약 의 경우 의료법 제3조(의료기관)의 규정에 따른 종합병원과 병원에서 직장 또는 개인 이 실시한 건강진단서 사본 등 '
 '건강상태를 판단할 수 있는 자료로 건강진단을 대신할 수 있습니다'),
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
