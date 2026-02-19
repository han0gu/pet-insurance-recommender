from langchain_core.documents import Document

chunk = Document(
    page_content=('② 계약자 또는 피보험자가 정당한 이유 없이 제1항의 의무를 이행하지 않았을 때에는 제 3조(보상하는 손해)에 의한 손해에서 다음의 '
 '금액을 뺍니다.\n'
 '1. 제1항 제1호의 경우에는 그 노력을 하였더라면 손해를 방지 또는 경감할 수 있었던 금액 2. 제1항 제2호의 경우에는 제3자로부터 '
 '손해의 배상을 받을 수 있었던 금액 3. 제1항 제3호의 경우에는 소송비용(중재 또는 조정에 관한 비용 포함) 및 변호사비용 과 회사의 '
 '동의를 받지 않은 행위로 증가된 손해\n'
 '제12조(손해배상청구에 대한 회사의 해결)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 26},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000161',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
