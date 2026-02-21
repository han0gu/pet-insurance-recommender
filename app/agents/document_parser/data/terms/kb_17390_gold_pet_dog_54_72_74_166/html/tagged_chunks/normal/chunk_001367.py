from langchain_core.documents import Document

chunk = Document(
    page_content=("회사와 계약자간에 사전에 합의가 있을 경우에 적용합니다.</h1><p id='15' data-category='paragraph' "
 "style='font-size:14px'>제2조(보장특약의 자동갱신)<br>\uf000 이 보장특약의 보험기간은 갱신전 보장특약의 "
 '보험기간으로 합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001367',
              'chunk_char_len': 151,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
