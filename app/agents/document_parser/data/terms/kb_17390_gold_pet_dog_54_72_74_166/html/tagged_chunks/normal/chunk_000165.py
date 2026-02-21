from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>등)</p><br><p id='205' data-category='paragraph' "
 "style='font-size:16px'>\uf000 회사는 계약자가 청약할 때에 계약자에게 약관의 중요한 내용을 설명하여야 "
 '하며,<br>청약 후에 다음 각 호의 방법 중 계약자가 원하는 방법을 확인하여 지체 없이 약관<br>및 계약자 보관용 청약서를 제공하여 '
 '드립니다'),
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
 'indexing': {'chunk_id': 'chunk_000165',
              'chunk_char_len': 216,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
