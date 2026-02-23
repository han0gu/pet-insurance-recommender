from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하여야 하며 계약자에 따라\n'
 '- 다르게 해석하지 않습니다.\n'
 '- \uf000 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게 유리하게 해석합니다.\n'
 '- \uf000 회사는 보험금을 지급하지 않는 사유 및 보상하지 않는 손해 등 계약자나 피보험\n'
 '- 자에게 불리하거나 부담을 주는 내용은 확대하여 해석하지 않습니다.\n'
 '| 용 어 풀 이 | 신의성실의 원칙 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000203',
              'chunk_char_len': 233,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
