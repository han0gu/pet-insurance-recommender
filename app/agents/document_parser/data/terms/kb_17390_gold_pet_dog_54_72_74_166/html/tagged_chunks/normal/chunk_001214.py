from langchain_core.documents import Document

chunk = Document(
    page_content=("id='11' data-category='paragraph' style='font-size:14px'>124 KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01)</p><br><p id='12' data-category='paragraph' "
 "style='font-size:14px'>\uf000 회사는 제1항에 따라 위험이 감소된 경우에는 그 차액보험료를 돌려드리며, "
 "위험</p><br><h1 id='13' style='font-size:14px'>이 증가된 경우에는 통지를 받은 날부터 1개월 이내에 "
 '보험료의 증액을'),
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
 'indexing': {'chunk_id': 'chunk_001214',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
