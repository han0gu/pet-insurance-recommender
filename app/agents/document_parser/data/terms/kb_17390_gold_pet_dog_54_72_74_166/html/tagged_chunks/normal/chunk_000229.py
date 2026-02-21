from langchain_core.documents import Document

chunk = Document(
    page_content=('납입하여야 하며, 회사는 계약자가 보 제28조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)</p><br><p '
 "id='40' data-category='paragraph' style='font-size:14px'>험료를 납입한 경우에는 영수증을 "
 '발행하여 드립니다'),
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
 'indexing': {'chunk_id': 'chunk_000229',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
