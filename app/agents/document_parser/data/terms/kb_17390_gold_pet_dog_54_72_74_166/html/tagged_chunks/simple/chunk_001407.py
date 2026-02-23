from langchain_core.documents import Document

chunk = Document(
    page_content=(". 소득세법 제59조의4(특별세액공제) 제1항 제2호에 따라 보험료가 특별세액공<br>제의 대상이 되는 보험</p><h1 id='70' "
 "style='font-size:14px'>∙</h1><br><p id='71' data-category='paragraph' "
 "style='font-size:14px'>소득세법 제59조의 4(특별세액공제)</p><br><p id='72' "
 "data-category='paragraph' style='font-size:14px'>근로소득이 있는 거주자(일용근로자는 제외한다"),
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
 'indexing': {'chunk_id': 'chunk_001407',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
