from langchain_core.documents import Document

chunk = Document(
    page_content=('의하여 제1조(보험금의 지급사유)<br>에서 정한 지급사유의 치료가 필요하다고 인정한 경우로서 자택 등에서 치료가 곤<br>란하여 '
 '의료기관에 입실하여 의사의 관리하에 치료에 전념하는 것을 말합니다.<br>\uf000 제1항의 "의료기관"이라 함은 의료법 '
 "제3조(의료기관) 제2항에서 정한 국내의 병</p><br><p id='1' data-category='paragraph' "
 "style='font-size:14px'>원이나 의원 또는 국외의 의료관련법에서 정한 의료기관을 말합니다.</p><p id='2'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000700',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
