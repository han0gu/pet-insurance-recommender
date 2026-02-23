from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 피보험자가 의무보험에 가입하여야 함에도 불구하고 가입하지 않은 경우에는 그\n'
 '- 가 가입했더라면 의무보험에서 보상했을 금액을 제1항의 "의무보험에서 보상하는\n'
 '# 금액"으로 봅니다.# 제11조(보험금의 분담)\uf000 회사는 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 '
 '계약(공제\n'
 '계약을 포함합니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하\n'
 '여 각각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손'),
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
 'indexing': {'chunk_id': 'chunk_000680',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
