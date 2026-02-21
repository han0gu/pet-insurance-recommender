from langchain_core.documents import Document

chunk = Document(
    page_content=("id='228' data-category='paragraph' style='font-size:16px'>∙</p><br><p "
 "id='229' data-category='paragraph' style='font-size:16px'>∙</p><br><p "
 "id='230' data-category='paragraph' style='font-size:16px'>∙</p><br><p "
 "id='231' data-category='list' style='font-size:16px'>위험보장사항 및 각각의 "
 '보험료<br>별<br>청약의 철회에 관한'),
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
 'indexing': {'chunk_id': 'chunk_000180',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
