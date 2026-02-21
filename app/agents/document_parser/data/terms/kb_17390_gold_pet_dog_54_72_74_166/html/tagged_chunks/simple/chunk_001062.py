from langchain_core.documents import Document

chunk = Document(
    page_content=("금쪽같은 펫보험(강아지)(무배당)(26.01) 115</p><br><p id='35' data-category='paragraph' "
 "style='font-size:18px'>- 115 -</p><h1 id='36' style='font-size:14px'>\uf000 "
 "제1항 제4호의 사고증명서는 수의사법 제12조(진단서 등)에서 규정한 내용에 따라</h1><br><table id='37' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>국내의</td><td>동물병원에서 수의사에 "
 '의해 발급한'),
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
 'indexing': {'chunk_id': 'chunk_001062',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
