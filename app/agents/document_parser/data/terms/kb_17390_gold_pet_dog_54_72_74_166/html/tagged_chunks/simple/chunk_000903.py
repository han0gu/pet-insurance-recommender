from langchain_core.documents import Document

chunk = Document(
    page_content=('체납할 경우 국세 기본법<br>병<br>및 지방세법에 의하여 체납된 세금에 대하여 가산금 징수, 독촉장 발부 및 재산<br>압류 등의 '
 '집행을 하는 것을 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000903',
              'chunk_char_len': 89,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
