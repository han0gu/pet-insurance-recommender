from langchain_core.documents import Document

chunk = Document(
    page_content=("하나에 해당하는 보험, 보증, 공제의 보험료, 보증료,</h1><br><p id='79' data-category='list' "
 "style='font-size:16px'>공제료 중 기획재정부령으로 정하는 것을 말한다.<br>1"),
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
 'indexing': {'chunk_id': 'chunk_001412',
              'chunk_char_len': 125,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
