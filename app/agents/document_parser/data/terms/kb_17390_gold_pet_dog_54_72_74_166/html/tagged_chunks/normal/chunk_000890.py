from langchain_core.documents import Document

chunk = Document(
    page_content=('납입이 연체되는 경우 납입최고(독촉)와 특별약관의 해지)에 따<br>라 특별약관이 해지되었으나 해약환급금을 받지 않은 경우(보험계약대출 '
 '등에 따<br>라 해약환급금이 차감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포<br>함합니다) 계약자는 해지된 날부터 3년 '
 '이내에 회사가 정한 절차에 따라 특별약관<br>의 부활(효력회복)을 청약할 수 있습니다'),
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
 'indexing': {'chunk_id': 'chunk_000890',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
