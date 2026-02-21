from langchain_core.documents import Document

chunk = Document(
    page_content=('② 계약자 또는 피보험자가 정당한 이유 없이 제1항의 의무를 이행하지 않았을 때에는\n'
 '제3조(보상하는 손해)에 의한 손해에서 다음의 금액을 뺍니다.- 1. 제1항 제1호의 경우에는 그 노력을 하였더라면 손해를 방지 또는 '
 '경감할 수 있었\n'
 '- 던 금액\n'
 '- 2. 제1항 제2호의 경우에는 제3자로부터 손해의 배상을 받을 수 있었던 금액\n'
 '- 3. 제1항 제3호의 경우에는 소송비용(중재 또는 조정에 관한 비용 포함) 및 변호사비\n'
 '- 용과 회사의 동의를 받지 않은 행위에 의하여 증가된 손해'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000496',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
