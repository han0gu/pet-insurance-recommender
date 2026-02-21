from langchain_core.documents import Document

chunk = Document(
    page_content=('. 병<br>\uf000 회사가 제1항의 절차에 협조하거나 대행하는 경우에는 피보험자는 회사의 요청에<br>따라 협력해야 하며, '
 '피보험자가 정당한 이유없이 협력하지 않을 경우에는 그로<br>말미암아 늘어난 손해에 대해서 보상하지 않습니다.<br>\uf000 회사는 '
 "다음의 경우에는 제1항의 절차를 대행하지 않습니다. 상</p><br><p id='212' data-category='paragraph' "
 "style='font-size:16px'>1"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001193',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
