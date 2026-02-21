from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피해자로부터 손해배상책임에 관한 소송을 제기 받았을 경우<br>3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변 '
 '제<br>\uf000 계약자 또는 피보험자가 제1항 각호의 통지를 게을리하여 손해가 증가된 때에는 도<br>4'),
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
 'indexing': {'chunk_id': 'chunk_001157',
              'chunk_char_len': 126,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
