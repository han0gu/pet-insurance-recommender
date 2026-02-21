from langchain_core.documents import Document

chunk = Document(
    page_content=(". 피보험자가 정당한 이유없이 협력하지 않을 때</p><br><p id='215' data-category='paragraph' "
 "style='font-size:16px'>\uf000 회사가 제1항의 절차를 대행하는 경우에는, 피보험자에 대하여 보상책임을 "
 '지는<br>한도 내에서, 가압류나 가집행을 면하기 위한 공탁금을 피보험자에게 대부할 수<br>있으며 이에 소요되는 비용을 보상합니다'),
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
 'indexing': {'chunk_id': 'chunk_001195',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
