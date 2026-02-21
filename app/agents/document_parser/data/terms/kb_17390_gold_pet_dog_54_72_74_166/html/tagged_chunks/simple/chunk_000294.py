from langchain_core.documents import Document

chunk = Document(
    page_content=('하나에 해당하는<br>자 중 2인이내에서 보험금의 대리청구인(이하, "지정대리청구인"이라 합니다)을<br>지정할 수 있으며, 2인을 '
 '지정대리청구인으로 지정시 대표대리인을 지정해야 합니<br>다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000294',
              'chunk_char_len': 107,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
