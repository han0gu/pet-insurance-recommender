from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약 상\n'
 '을 포함합니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각 해\n'
 '각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손해를| 보상합니다. 피보험자가 부담한 치료비 | × | 다른 |\n'
 '| --- | --- | --- |\n'
 '| 보상합니다. 피보험자가 부담한 치료비 | × | 계약이 없을 때 이 계약의 지급보험금 다른계약이 없는 것으로 하여 각각 계산한 '
 '지급보험금의 합계액 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000477',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
