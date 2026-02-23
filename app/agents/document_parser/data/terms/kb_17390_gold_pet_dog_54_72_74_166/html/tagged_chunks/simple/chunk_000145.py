from langchain_core.documents import Document

chunk = Document(
    page_content=("계약의 해지)에서 정한 계약의 해지가 발생하지 않</p><br><h1 id='174' style='font-size:14px'>은 경우를 "
 "말합니다.</h1><br><p id='175' data-category='paragraph' "
 "style='font-size:14px'>\uf000 제29조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))에서 정한 "
 "계약</p><br><p id='176' data-category='paragraph' style='font-size:14px'>의 부활이 "
 '이루어진 경우 부활을 청약한 날을 제5항의'),
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
 'indexing': {'chunk_id': 'chunk_000145',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
