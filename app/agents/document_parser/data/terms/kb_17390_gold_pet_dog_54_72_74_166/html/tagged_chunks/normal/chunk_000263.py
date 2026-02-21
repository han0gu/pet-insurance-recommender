from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>등</p><h1 id='68' style='font-size:14px'>제31조(계약자의 "
 "임의해지 및 피보험자의 서면동의</h1><br><p id='69' data-category='paragraph' "
 "style='font-size:14px'>철회)</p><br><p id='70' data-category='paragraph' "
 "style='font-size:14px'>\uf000 계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지할 수 있으며, 이 경우 "
 '회<br>사는 제34조(해약환급금) 제1항에'),
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
 'indexing': {'chunk_id': 'chunk_000263',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
