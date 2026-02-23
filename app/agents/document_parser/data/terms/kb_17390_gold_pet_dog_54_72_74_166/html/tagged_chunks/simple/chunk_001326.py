from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이륜인 자동차에 측차를 붙인 자동차</p><br><p id='189' data-category='paragraph' "
 "style='font-size:14px'>132 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='190' "
 "data-category='paragraph' style='font-size:14px'>2"),
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
 'indexing': {'chunk_id': 'chunk_001326',
              'chunk_char_len': 192,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
