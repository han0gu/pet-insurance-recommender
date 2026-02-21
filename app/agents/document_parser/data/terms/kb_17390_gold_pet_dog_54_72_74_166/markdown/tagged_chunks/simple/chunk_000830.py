from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '# 별표1 장해분류표# \uf000 총칙- 1. 장해의 정의\n'
 '- 1) ‘장해’라 함은 상해 또는 질병에 대하여 치유된 후 신체에 남아 있는 영구\n'
 '- 적인 정신 또는 육체의 훼손상태 및 기능상실 상태를 말한다. 다만, 질병과\n'
 '- 부상의 주증상과 합병증상 및 이에 대한 치료를 받는 과정에서 일시적으로\n'
 '- 나타나는 증상은 장해에 포함되지 않는다.\n'
 '- 2) ‘영구적’이라 함은 원칙적으로 치유하는 때 장래 회복할 가망이 없는 상태\n'
 '- 로서 정신적 또는 육체적 훼손상태임이 의학적으로 인정되는 경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000830',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
