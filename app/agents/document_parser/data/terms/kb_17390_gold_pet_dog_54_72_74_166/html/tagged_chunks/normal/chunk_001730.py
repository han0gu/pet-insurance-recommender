from langchain_core.documents import Document

chunk = Document(
    page_content=('colspan="2"><table><thead><tr><td>부목-장상지(상완으로부터 '
 '수부까지)</td><td>T6151</td></tr></thead><tbody><tr><td>부목-단상지(전완으로부터 '
 '수부까지)</td><td>T6152</td></tr><tr><td>부목-장하지(대퇴로부터 '
 '족부까지)</td><td>T6153</td></tr><tr><td>부목-단상지(하퇴로부터'),
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
 'indexing': {'chunk_id': 'chunk_001730',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
