from langchain_core.documents import Document

chunk = Document(
    page_content=(". 피보험자의 3촌 이내의 친족<br>\uf000 제1항의 규정에 의하여 회사가 이 특별약관의 보험금을</p><br><p id='212' "
 "data-category='paragraph' style='font-size:16px'>지정대리청구인에게 지급</p><br><p "
 "id='213' data-category='paragraph' style='font-size:16px'>한 경우에는 그 이후 이 "
 "특별약관의 보험금 청구를 받더라도 회사는 이를 지급하<br>지 않습니다.</p><br><p id='214'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001343',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
