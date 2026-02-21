from langchain_core.documents import Document

chunk = Document(
    page_content=("회사는 이 특별약관이 부가된</h1><br><p id='93' data-category='list' "
 "style='font-size:14px'>제1항 제1호에 해당하는 장애인전용보험으로 전환하여 드립니다.<br>\uf000 제1항에 따라 "
 "전환대상계약이 장애인전용보험으로 전환된 후부터 납입된 전환대상 약</p><br><p id='94' "
 "data-category='paragraph' style='font-size:20px'>전환대상계약을 소득세법 제59조의4(특별세액공제) "
 "성특</p><br><p id='95'"),
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
 'indexing': {'chunk_id': 'chunk_001427',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
