from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 "천식지속상태"의 진단확정은 의료법 제3조(의료기관)에서 정한 국내의 병원이나\n'
 '- 의원 또는 국외의 의료관련법에서 정한 의료기관의 의사(치과의사 제외) 면허를\n'
 '- 가진 자에 의하여 내려져야 합니다. 또한, 회사가 "천식지속상태"의 조사나 확인\n'
 '- 을 위하여 필요하다고 인정하는 경우에는 검사결과, 진료기록부의 사본 제출을'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000400',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
