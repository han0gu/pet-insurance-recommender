from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 계약자, 피보험자 또는 보험수익자가 보험금 청구에 관한 서류에 고의로 사실과\n'
 '다른 것을 기재하였거나 그 서류 또는 증거를 위조 또는 변조한 경우. 다만, 이\n'
 '미 보험금 지급사유가 발생한 경우에는 이에 대한 보험금은 지급합니다.- \n'
 '\uf000 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취지를 계약자에게 통지하고'),
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
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
