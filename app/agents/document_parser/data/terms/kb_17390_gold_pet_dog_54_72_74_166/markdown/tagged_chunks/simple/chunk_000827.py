from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 제1항에서 지정한 특정질병의 합병증으로 인해 발생한 특정질병이외의 질병\n'
 '- 으로 보험계약에서 정한 보험금의 지급사유가 발생한 경우\n'
 '- 2. 상해를 직접적인 원인으로 하여 보험금의 지급사유가 발생한 경우\n'
 '- 3. 제1항에서 지정한 특정질병으로 인하여 사망하여 보험금의 지급사유가 발생\n'
 '- 한 경우\n'
 '- \uf000 해당 반려동물에게 보험사고가 발생했을 경우, 그 사고가 특정질병을 직접적인 원\n'
 '- 인으로 발생한 사고인가 아닌가는 수의사의 진단서와 의견을 주된 판단자료로 결\n'
 '- 정합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000827',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
